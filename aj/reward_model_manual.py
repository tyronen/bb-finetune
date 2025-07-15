import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os


# ========== HYPERPARAMETERS ==========
MODEL_PATH = "qwen-sft-checkpoint/merged"
TOKENIZER_PATH = "qwen-sft-checkpoint/checkpoint-3000"
BATCH_SIZE = 16
MAX_LENGTH = 512
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "qwen-reward-manual-checkpoint"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# =====================================


class ValueHead(nn.Module):
    """Simple linear value head for reward modeling."""
    def __init__(self, hidden_size):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # Expecting (batch_size, seq_len, hidden_size)
        # Take final token's hidden state for each example
        if hidden_states.dim() == 3:
            x = hidden_states[:, -1, :]
        else:
            x = hidden_states
        return self.value_head(x)  # shape: (batch, 1)ch, 1)


class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # store the LM backbone
        self.base_model = base_model
        # simple linear head on top of the last hidden state's final token
        self.value_head = ValueHead(base_model.config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        # single forward through the backbone
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state        # (batch, seq_len, hidden)
        pooled      = last_hidden[:, -1, :]            # (batch, hidden)
        rewards     = self.value_head(pooled).squeeze(-1)  # (batch,)
        return rewards


class RewardComparisonDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        # Concatenate prompt + response for both pairs
        chosen_enc = self.tokenizer(
            prompt + chosen,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            prompt + rejected,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }

def reward_model_loss(chosen_rewards, rejected_rewards):
    # chosen_rewards, rejected_rewards: (batch,)
    diff = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(diff).mean()
    return loss

def evaluate(model, dataloader, DEVICE):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            chosen_ids = batch["chosen_input_ids"].to(DEVICE)
            chosen_mask = batch["chosen_attention_mask"].to(DEVICE)
            rejected_ids = batch["rejected_input_ids"].to(DEVICE)
            rejected_mask = batch["rejected_attention_mask"].to(DEVICE)

            all_ids  = torch.cat([batch["chosen_input_ids"],
                                  batch["rejected_input_ids"]], dim=0).to(DEVICE)
            all_mask = torch.cat([batch["chosen_attention_mask"],
                                  batch["rejected_attention_mask"]], dim=0).to(DEVICE)
            all_rewards = model(all_ids, all_mask)
            chosen_rewards, rejected_rewards = all_rewards.split(chosen_ids.size(0), dim=0)
            loss = reward_model_loss(chosen_rewards, rejected_rewards)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(dataloader)
    return avg_val_loss

def collate_fn(batch):
    # batch is a list of dicts with four tensor fields
    chosen_ids   = [x["chosen_input_ids"]   for x in batch]
    chosen_mask  = [x["chosen_attention_mask"] for x in batch]
    rejected_ids = [x["rejected_input_ids"]  for x in batch]
    rejected_mask= [x["rejected_attention_mask"] for x in batch]

    # pad to longest in this mini‐batch
    chosen_ids   = pad_sequence(chosen_ids,  batch_first=True, padding_value=tokenizer.pad_token_id)
    chosen_mask  = pad_sequence(chosen_mask, batch_first=True, padding_value=0)
    rejected_ids = pad_sequence(rejected_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    rejected_mask= pad_sequence(rejected_mask,batch_first=True, padding_value=0)

    return {
        "chosen_input_ids":   chosen_ids,
        "chosen_attention_mask": chosen_mask,
        "rejected_input_ids":  rejected_ids,
        "rejected_attention_mask": rejected_mask,
    }

# ====== DATA LOADING ======
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)


train_dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="train")
val_dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="valid1[:2000]")

train_dataset = RewardComparisonDataset(train_dataset, tokenizer, max_length=MAX_LENGTH)
val_dataset = RewardComparisonDataset(val_dataset, tokenizer, max_length=MAX_LENGTH)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)


# ====== TRAINING SETUP ======
base_model   = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
reward_model = RewardModel(base_model).to(DEVICE)

for p in reward_model.base_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(
    reward_model.value_head.parameters(), lr=LEARNING_RATE
)

scaler = GradScaler()

# ====== TRAINING LOOP ======
for epoch in range(NUM_EPOCHS):
    reward_model.train()
    total_loss = 0
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad()
        chosen_ids = batch["chosen_input_ids"].to(DEVICE)
        chosen_mask = batch["chosen_attention_mask"].to(DEVICE)
        rejected_ids = batch["rejected_input_ids"].to(DEVICE)
        rejected_mask = batch["rejected_attention_mask"].to(DEVICE)

        with torch.cuda.amp.autocast():
            # 1) Merge along batch dimension: now size [2B, L]
            all_ids   = torch.cat([chosen_ids,   rejected_ids],   dim=0)
            all_mask  = torch.cat([chosen_mask,  rejected_mask],  dim=0)

            # 2) One forward pass: returns [2B] rewards
            all_rewards = reward_model(all_ids, all_mask)

            # 3) Split back into chosen vs rejected
            chosen_rewards, rejected_rewards = all_rewards.split(chosen_ids.size(0), dim=0)

            loss = reward_model_loss(chosen_rewards, rejected_rewards)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_dataloader)
    print(f"\nEpoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    # ====== VALIDATION LOOP ======
    avg_val_loss = evaluate(reward_model, val_dataloader, DEVICE)
    print(f"Val Loss: {avg_val_loss:.4f}")

# ====== SAVE MODEL ======
# Save the reward model (including value head) as a HuggingFace model
full_path = os.path.join(MODEL_SAVE_PATH, "full_reward_model.pt")
torch.save(reward_model.state_dict(), full_path)
print(f"Saved full reward model to {full_path}")

# Save the value head weights separately if needed (HF doesn't handle custom heads natively)
head_path = os.path.join(MODEL_SAVE_PATH, "value_head.pt")
torch.save(reward_model.value_head.state_dict(), head_path)
print(f"Saved value head to {head_path}")

# Save the tokenizer
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Model and tokenizer saved to {MODEL_SAVE_PATH}/")