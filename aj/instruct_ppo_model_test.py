import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # Only needed if you used LoRA/PEFT for PPO
import os

# ---- SETTINGS ----
sft_dir = "./qwen-sft-instruct-checkpoint/merged"
ppo_dir = "./qwen-ppo-rlhf-checkpoint/"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_new_tokens = 128

def generate_response(model, prompt, tokenizer, max_new_tokens=max_new_tokens):
    input_text = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Only return the assistant part
    if "Assistant:" in decoded:
        return decoded.split("Assistant:", 1)[-1].strip()
    else:
        return decoded

# ---- LOAD TOKENIZER ----
tokenizer = AutoTokenizer.from_pretrained(sft_dir, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ---- LOAD MODELS ----
sft_model = AutoModelForCausalLM.from_pretrained(sft_dir, trust_remote_code=True).to(device)
sft_model.eval()

ppo_model= AutoModelForCausalLM.from_pretrained(ppo_dir, trust_remote_code=True).to(device)
ppo_model.eval()

# ---- TEST PROMPTS ----
test_prompts = [
    "How do I make a cup of tea?",
    "Tell me a fun fact about space.",
    "What's the capital of Japan?",
    "Explain overfitting in machine learning.",
    "What are the ingredients for pancakes?"
]

print("\n=== QwenSFT Instruct vs RLHF PPO ===")
for prompt in test_prompts:
    print(f"\nUser: {prompt}")
    print("\n[SFT Model Response]\n", generate_response(sft_model, prompt, tokenizer))
    print("\n[RLHF PPO Model Response]\n", generate_response(ppo_model, prompt, tokenizer))

# ---- INTERACTIVE COMPARISON ----
while True:
    user_input = input("\nYou (or type 'exit'): ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break
    print("\n[SFT Model]:\n", generate_response(sft_model, user_input, tokenizer))
    print("\n[RLHF PPO Model]:\n", generate_response(ppo_model, user_input, tokenizer))
