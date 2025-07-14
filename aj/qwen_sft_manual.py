import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import evaluate
import math
from tqdm import tqdm
import os

class LoRALinear(nn.Module):
    def __init__(self, orig_linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.orig_linear = orig_linear
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        in_features = orig_linear.in_features
        out_features = orig_linear.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scaling = self.alpha / self.r

    def forward(self, x):
        result = self.orig_linear(x)
        lora_out = self.dropout(x) @ self.lora_A.T
        lora_out = lora_out @ self.lora_B.T
        return result + self.scaling * lora_out

class TLDRDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["input_ids"]),
        }

def add_lora_to_model(model, r=8, alpha=16, dropout=0.05):
    for name, module in model.named_modules():
        # Patch Q, K, V, O projections with LoRA
        if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            if isinstance(module, nn.Linear):
                parent = model
                for attr in name.split('.')[:-1]:
                    parent = getattr(parent, attr)
                orig_linear = getattr(parent, name.split('.')[-1])
                lora_linear = LoRALinear(orig_linear, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, name.split('.')[-1], lora_linear)
    return model

def tokenize_fn(sample):
    txt = sample["prompt"] + sample["label"]
    enc = tokenizer(
        txt,
        truncation=True,
        max_length=550,
        padding="max_length",
    )
    sample["input_ids"] = enc["input_ids"]
    sample["attention_mask"] = enc["attention_mask"]
    return sample


def evaluate_model(model, val_loader, tokenizer, rouge, device, max_new_tokens=100, num_batches=None):
    model.eval()
    predictions = []
    references = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Generate summaries
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Remove prompt from output
            pred_str = tokenizer.batch_decode(
                [gen[seq.shape[1]:] for gen, seq in zip(generated, input_ids)],
                skip_special_tokens=True
            )
            label_str = tokenizer.batch_decode(
                [lab[lab != tokenizer.pad_token_id] for lab in labels],
                skip_special_tokens=True
            )

            predictions.extend(pred_str)
            references.extend(label_str)          

    results = rouge.compute(predictions=predictions, references=references)
    model.train()  # restore train mode
    return results


# Load ROUGE for summarization evaluation
rouge = evaluate.load("rouge")


model_name = "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


model = add_lora_to_model(model, r=8, alpha=16, dropout=0.05)

for name, param in model.named_parameters():
    if 'lora_' not in name:
        param.requires_grad = False


# Load dataset (e.g., CarperAI TL;DR)
data_path = "CarperAI/openai_summarize_tldr"
dataset = load_dataset(data_path, split="train")
val_dataset = load_dataset(data_path, split="valid")

# Pre-tokenize dataset
tokenized_train_dataset = dataset.map(tokenize_fn)
tokenized_val_dataset = val_dataset.map(tokenize_fn)

# Create train and validation datasets
train_dataset = TLDRDataset(tokenized_train_dataset)
val_dataset = TLDRDataset(tokenized_val_dataset)

NUM_WORKERS = 16   # Try 8, 16, or even 24 if CPU usage/RAM is ok
BATCH_SIZE = 4     # Or higher if your GPU has VRAM to spare
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 2
LOG_INTERVAL = 100
EVAL_INTERVAL = 500

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

lora_params = [p for n, p in model.named_parameters() if p.requires_grad]

optimizer = optim.AdamW(lora_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

TOTAL_STEPS = NUM_EPOCHS * len(train_loader)
WARMUP_STEPS = int(0.06 * TOTAL_STEPS)
OPTIMIZER_STEPS = 0

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=TOTAL_STEPS,
)

model.train()

for epoch in range(NUM_EPOCHS):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, batch in pbar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if 'scheduler' in locals():
            scheduler.step()

        optimizer_steps += 1

        if optimizer_steps % EVAL_INTERVAL == 0:
            rouge_scores = evaluate_model(
                model, val_loader, tokenizer, rouge, device, max_new_tokens=100, num_batches=20  # eval on 20 batches for speed
            )
            print(f"Step {optimizer_steps} | Loss: {loss.item():.4f} | ROUGE: {rouge_scores}")
        
        if step % LOG_INTERVAL == 0:
            pbar.set_description(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")


output_dir = "./qwen-sft-manual-checkpoint"
os.makedirs(output_dir, exist_ok=True)

# Merge LoRA weights into base model
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            orig_weight = module.orig_linear.weight.data
            lora_update = module.scaling * (module.lora_B @ module.lora_A)
            module.orig_linear.weight.data = orig_weight + lora_update
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()
    return model

merged_model = merge_lora_weights(model)
merged_model.cpu()

# Save the full model (HuggingFace-compatible)
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

model = AutoModelForCausalLM.from_pretrained("./qwen-sft-manual-checkpoint", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./qwen-sft-manual-checkpoint", trust_remote_code=True)