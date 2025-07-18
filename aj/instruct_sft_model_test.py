import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---- SETTINGS ----
merged_dir = "./qwen-sft-instruct-checkpoint/merged"  # Update path as needed
base_model_name = "Qwen/Qwen3-0.6B-Base"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_new_tokens = 128

# ---- HELPER FUNCTION FOR GENERATION ----
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


# ---- LOAD MODEL + TOKENIZER ----
tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # For safety
model = AutoModelForCausalLM.from_pretrained(merged_dir, trust_remote_code=True).to(device)
model.eval()

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True).to(device)
base_model.eval()

# ---- TEST PROMPTS ----
test_prompts = [
    "How do I make a cup of tea?",
    "Tell me a fun fact about space.",
    "What's the capital of Japan?",
    "Explain overfitting in machine learning.",
    "What are the ingredients for pancakes?"
]

print("\n=== Qwen3 0.6B Base vs SFT Instruct Model Comparison ===")
for prompt in test_prompts:
    print(f"\nUser: {prompt}")
    base_resp = generate_response(base_model, prompt, tokenizer)
    sft_resp = generate_response(model, prompt, tokenizer)
    print("\n[Base Model Response]\n", base_resp)
    print("\n[SFT Model Response]\n", sft_resp)
    print("-" * 60)

# ---- INTERACTIVE COMPARISON ----
while True:
    user_input = input("\nYou (or type 'exit'): ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break
    base_out = generate_response(base_model, user_input, tokenizer)
    sft_out = generate_response(model, user_input, tokenizer)
    print("\n[Base Model]:\n", base_out)
    print("\n[SFT Model]:\n", sft_out)
    print("-" * 60)