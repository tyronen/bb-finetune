import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, default_data_collator
import random
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel
import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"

class TLDRDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=550):
        self.post_list = []
        dataset = load_dataset(train_path, split=split)
        for sample in dataset:
            self.post_list.append(sample["prompt"] + sample["label"])
        if "valid" in split:
            self.post_list = self.post_list[0:2000]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        txt = self.post_list[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


# Hyperparameters and output settings
output_dir = "./qwen-sft-checkpoint"
train_batch_size = 4  # reduced due to larger model size
gradient_accumulation_steps = 4  # helps simulate batch size 16
learning_rate = 1e-5
eval_batch_size = 1
eval_steps = 500
max_input_length = 550
save_steps = 1000
num_train_epochs = 1

set_seed(42)

# Load Qwen model/tokenizer

model_name = "Qwen/Qwen3-0.6B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.config.pad_token_id = tokenizer.pad_token_id


# Define the LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply the LoRA adapter to the base model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.to(device)
model.train()


data_path = "CarperAI/openai_summarize_tldr"

train_dataset = TLDRDataset(
    data_path,
    tokenizer,
    "train",
    max_length=max_input_length,
)

dev_dataset = TLDRDataset(
    data_path,
    tokenizer,
    "valid",
    max_length=max_input_length,
)

dataset = load_dataset(data_path, split='train')


# Load ROUGE for summarization evaluation
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    label_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    result = rouge.compute(predictions=pred_str, references=label_str)
    return result

# Optional: extract only the best predictions
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


training_args = TrainingArguments(
    auto_find_batch_size=True,
    output_dir=output_dir,
    dataloader_num_workers=4,
    dataloader_persistent_workers=True,
    dataloader_pin_memory=True,
    eval_strategy="steps",
    eval_accumulation_steps=1,
    learning_rate=learning_rate,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_checkpointing=False,
    fp16=True,
    adam_beta1=0.9,
    adam_beta2=0.95,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    optim="adamw_torch_fused",
    warmup_steps=100,
    eval_steps=eval_steps,
    save_steps=save_steps,
    max_steps=1500,  # Optional: control total training steps
    load_best_model_at_end=True,
    warmup_steps=100,
    logging_steps=50,
    report_to="none",  # disable W&B or others unless configured
    label_names=["labels"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(output_dir)


# Load the PEFT config to get base model path
peft_config = PeftConfig.from_pretrained(output_dir)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    trust_remote_code=True,
)

# Load trained PEFT model
model = PeftModel.from_pretrained(base_model, output_dir)
model = model.to(device)

# Option 1: Save PEFT model (LoRA weights only)
model.save_pretrained(output_dir)

# Option 2 (Optional): Merge LoRA weights into base model for export/inference
merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"{output_dir}/merged")

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(output_dir)