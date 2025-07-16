import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils

# --- Load tokenizer and dataset ---
model_name = utils.BASE
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

from datasets import load_dataset

data_path = "CarperAI/openai_summarize_tldr"
dataset = load_dataset(data_path, split="valid")
# dataset = load_dataset(data_path, split="train")

device = utils.get_device()

# --- Prepare input from dataset[0] ---
input_text = dataset[0]["prompt"]
target_summary = dataset[0]["label"]

inputs = tokenizer(
    input_text, return_tensors="pt", padding=True, truncation=True, max_length=550
)
inputs = {k: v.to(device) for k, v in inputs.items()}

# --- Try loading merged model first ---
merged_model_dir = utils.SFT_DIR
if os.path.exists(os.path.join(merged_model_dir, "model.safetensors")):
    print("Using merged SFT model for inference.")
    sft_model = AutoModelForCausalLM.from_pretrained(
        merged_model_dir, trust_remote_code=True
    )
    sft_model = sft_model.to(device)
    sft_model.eval()
else:
    print("Merged model not found. Falling back to PEFT LoRA adapter weights.")
    from peft import PeftModel, PeftConfig

    sft_model_dir = "data/sft-separate"
    peft_config = PeftConfig.from_pretrained(sft_model_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path, trust_remote_code=True
    )
    sft_model = PeftModel.from_pretrained(base_model, sft_model_dir)
    sft_model = sft_model.to(device)
    sft_model.eval()

# --- Load Base model (no SFT) ---
base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
base_model = base_model.to(device)
base_model.eval()

# --- Generate with SFT model ---
with torch.no_grad():
    sft_outputs = sft_model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    sft_summary = tokenizer.decode(
        sft_outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

# --- Generate with Base model ---
with torch.no_grad():
    base_outputs = base_model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    base_summary = tokenizer.decode(
        base_outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

# --- Print comparison ---
print("\n--- Original Text ---\n", input_text)
print("\n--- Target Summary (Ground Truth) ---\n", target_summary)
print("\n--- SFT Model Prediction ---\n", sft_summary)
print("\n--- Base Model Prediction ---\n", base_summary)
