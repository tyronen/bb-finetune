
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead, RewardTrainer, RewardConfig
from datasets import load_dataset
import torch

class RewardModelWithDictOutput(AutoModelForCausalLMWithValueHead):
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Call the parent to get reward output
        rewards = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        # Most TRL reward trainers expect a dict
        # If rewards is a tuple, extract the scalar
        if isinstance(rewards, tuple):
            rewards = rewards[0]
        return {"logits": rewards}

class RewardModelTrainer(RewardTrainer):
    def create_optimizer(self):
        # Only pass value head parameters to optimizer
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if "v_head" in n]}
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
        )
        return self.optimizer

# Define accuracy metric
def compute_reward_accuracy(eval_preds):
    # eval_preds is a tuple: (scores, _)
    # scores: shape (batch_size, 2) or two separate arrays
    chosen_scores, rejected_scores = eval_preds
    accuracy = (chosen_scores > rejected_scores).mean()
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


model_path = "./qwen-sft-checkpoint/merged"
tokenizer_path = "./qwen-sft-checkpoint/checkpoint-3000"

# 1. Load tokenizer and model (with value head)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
reward_model = RewardModelWithDictOutput.from_pretrained(model_path)

# 3. Prepare train_dataset
train_dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="train")
val_dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="valid1[:2000]")

train_dataset = train_dataset.map(tokenize_pair, batched=True)
val_dataset   = val_dataset.map(tokenize_pair, batched=True)


def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)")
    
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

# 4. Train using RewardTrainer (from TRL)
trainer = RewardModelTrainer(
    model=reward_model,
    args=reward_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_reward_accuracy,
    processing_class=tokenizer,
)

trainer.train()