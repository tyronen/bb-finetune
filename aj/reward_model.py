
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset
import torch
import os
import numpy as np


MODEL_PATH = "./qwen-sft-checkpoint/merged"
TOKENIZER_PATH = "./qwen-sft-checkpoint/checkpoint-3000"
SAVE_PATH = "reward-model-checkpoint"
os.makedirs(SAVE_PATH, exist_ok=True)


# Define accuracy metric
def compute_reward_accuracy(eval_preds):
    # eval_preds is a tuple: (predictions, labels)
    # predictions has shape (num_examples, 2), cols = [chosen_score, rejected_score]
    preds, _ = eval_preds
    # Ensure it’s a NumPy array
    preds = np.asarray(preds)
    chosen_scores   = preds[:, 0]
    rejected_scores = preds[:, 1]
    accuracy = float((chosen_scores > rejected_scores).mean())
    return {"accuracy": accuracy}


def tokenize_pair(batch):
    chosen_list = [p + c for p, c in zip(batch["prompt"], batch["chosen"])]
    rejected_list = [p + r for p, r in zip(batch["prompt"], batch["rejected"])]
    tok_c = tokenizer(chosen_list, truncation=True, padding="max_length", max_length=512)
    tok_r = tokenizer(rejected_list, truncation=True, padding="max_length", max_length=512)
    return {
        "input_ids_chosen": tok_c["input_ids"],
        "attention_mask_chosen": tok_c["attention_mask"],
        "input_ids_rejected": tok_r["input_ids"],
        "attention_mask_rejected": tok_r["attention_mask"],
    }

def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)")


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token
reward_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=1,
    trust_remote_code=True,
)
reward_model.config.pad_token_id = tokenizer.pad_token_id


# 3. Prepare train_dataset
train_dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="train")
val_dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="valid1[:2000]")

train_dataset = train_dataset.map(tokenize_pair, batched=True)
val_dataset   = val_dataset.map(tokenize_pair, batched=True)


# 3) Freeze all but the score head
for name, param in reward_model.named_parameters():
    if name.startswith("score."):
        param.requires_grad = True
    else:
        param.requires_grad = False


print_trainable_params(reward_model)


# Fix for TRL's RewardTrainer bug
reward_model.warnings_issued = {}


reward_config = RewardConfig(
    output_dir="./reward_model_output",
    eval_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    fp16=True,
    logging_steps=50,
    report_to="none",
    remove_unused_columns=False,
)

trainer = RewardTrainer(
    model=reward_model,
    args=reward_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_reward_accuracy,
    processing_class=tokenizer,
)

trainer.train()


# Save the reward model (including value head) as a HuggingFace model
trainer.save_model(SAVE_PATH)

# Save the value head weights separately if needed (HF doesn't handle custom heads natively)
torch.save(
    reward_model.score.state_dict(),
    os.path.join(SAVE_PATH, "score_head.pt")
)

# Save the tokenizer
tokenizer.save_pretrained(SAVE_PATH)
print(f"Model and tokenizer saved to {SAVE_PATH}/")