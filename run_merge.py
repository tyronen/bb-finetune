from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils

# Load the base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(utils.BASE)
tokenizer = AutoTokenizer.from_pretrained(utils.SFT_DIR)
tokenizer.pad_token = tokenizer.eos_token

# Resize embeddings to match your tokenizer (very important!)
base_model.resize_token_embeddings(len(tokenizer))

# Load the LoRA adapter (from your SFT directory)
model = PeftModel.from_pretrained(base_model, utils.SFT_DIR)

# Merge the adapter weights into the model
merged_model = model.merge_and_unload()

# Save the merged model and tokenizer to a new directory
MERGED_DIR = utils.SFT_DIR + "-merged"
merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print(f"Merged model and tokenizer saved to {MERGED_DIR}")
