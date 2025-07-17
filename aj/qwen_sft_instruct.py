import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, default_data_collator
import random
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
import evaluate


class OASSTDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=1024):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prompt = self.pairs[idx]['prompt']
        response = self.pairs[idx]['response']
        # You can format as "User: <prompt>\nAssistant: <response>"
        text = f"User: {prompt}\nAssistant: {response}"
        tokenized = self.tokenizer(
            text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )
        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)
        labels = input_ids.clone()  # For Causal LM
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def build_prompt_response_pairs(dataset):
    # Build message_id -> item map for fast lookup
    id_to_msg = {msg['message_id']: msg for msg in dataset}
    pairs = []

    for msg in dataset:
        if msg["role"] == "assistant":
            parent_id = msg.get("parent_id")
            if not parent_id:
                continue
            parent = id_to_msg.get(parent_id)
            if parent and parent["role"] == "prompter":
                prompt = parent["text"]
                response = msg["text"]
                pairs.append({"prompt": prompt, "response": response})

    return pairs


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")

def compute_metrics(eval_preds):
    label_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    bertscore_result = bertscore.compute(predictions=pred_str, references=label_str, lang="en")
    bleu_result = bleu.compute(predictions=pred_str, references=[[x] for x in label_str])
    # BERTScore returns dict with precision, recall, f1 lists; take means for reporting
    return {
        "bertscore_precision": sum(bertscore_result['precision']) / len(bertscore_result['precision']),
        "bertscore_recall": sum(bertscore_result['recall']) / len(bertscore_result['recall']),
        "bertscore_f1": sum(bertscore_result['f1']) / len(bertscore_result['f1']),
        "bleu": bleu_result["bleu"],
    }

def print_sample_generations(model, tokenizer, n=5):
    model.eval()
    for i in range(n):
        sample = random.choice(val_pairs)
        prompt = sample["prompt"]
        input_text = f"User: {prompt}\nAssistant:"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.95)
        print("\n---")
        print("Prompt:", prompt)
        print("Reference:", sample["response"])
        print("Model:", tokenizer.decode(output[0], skip_special_tokens=True))


# Hyperparameters and output settings
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "./qwen-sft-instruct-checkpoint"
train_batch_size = 4  # reduced due to larger model size
gradient_accumulation_steps = 4  # helps simulate batch size 16
learning_rate = 1e-5
eval_batch_size = 1
eval_steps = 10
max_input_length = 1024
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


train_data = load_dataset("OpenAssistant/oasst1", split="train")
val_data = load_dataset("OpenAssistant/oasst1", split="validation[:100]")

train_pairs = build_prompt_response_pairs(train_data)
val_pairs = build_prompt_response_pairs(val_data)

print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

train_dataset = OASSTDataset(train_pairs, tokenizer, max_length=1024)
val_dataset = OASSTDataset(val_pairs, tokenizer, max_length=1024)


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
    load_best_model_at_end=True,
    logging_steps=50,
    report_to="none",  # disable W&B or others unless configured
    label_names=["labels"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
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


merged_model.to(device)
merged_model.eval()

test_prompt = "How do I make a cup of tea?"
input_text = f"User: {test_prompt}\nAssistant:"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
output = merged_model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))


print_sample_generations(merged_model, tokenizer)