import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os


# ========== HYPERPARAMETERS ==========
MODEL_PATH = "qwen-sft-checkpoint/merged"
TOKENIZER_PATH = "qwen-sft-checkpoint/checkpoint-3000"
BATCH_SIZE = 4
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
        return self.value_head(hidden_states[:, -1, :])  # shape: (batch, 1)


class RewardModel(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # Only a single tensor of shape (batch, seq_len, hidden)
        last_hidden = outputs.last_hidden_state  
        # Pool to (batch, hidden)
        pooled = last_hidden[:, -1, :]
        # Predict reward
        return self.value_head(pooled).squeeze(-1)


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

            chosen_rewards = model(chosen_ids, chosen_mask)
            rejected_rewards = model(rejected_ids, rejected_mask)
            loss = reward_model_loss(chosen_rewards, rejected_rewards)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(dataloader)
    return avg_val_loss


# ====== DATA LOADING ======
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)


train_dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="train")
val_dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="valid1[:2000]")

train_dataset = RewardComparisonDataset(train_dataset, tokenizer, max_length=MAX_LENGTH)
val_dataset = RewardComparisonDataset(val_dataset, tokenizer, max_length=MAX_LENGTH)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


# ====== TRAINING SETUP ======
reward_model = RewardModel(model).to(DEVICE)

for p in reward_model.model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(
    reward_model.value_head.parameters(), lr=LEARNING_RATE
)


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

        chosen_rewards = reward_model(chosen_ids, chosen_mask)
        rejected_rewards = reward_model(rejected_ids, rejected_mask)
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
reward_model.base_model.save_pretrained(MODEL_SAVE_PATH)
reward_model.value_head.cpu()  # Move to CPU to avoid DEVICE mismatch

# Save the value head weights separately if needed (HF doesn't handle custom heads natively)
torch.save(reward_model.value_head.state_dict(), os.path.join(MODEL_SAVE_PATH, "value_head.pt"))

# Save the tokenizer
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Model and tokenizer saved to {MODEL_SAVE_PATH}/")