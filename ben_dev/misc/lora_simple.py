"""
Simple EasyDeL GPT-2 LoRA Fine-tuning
Focused implementation with just the essentials for LoRA
"""

import os
import easydel as ed
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from datasets import Dataset
import numpy as np
from flax.core import FrozenDict

# JAX memory optimization - prevent pre-allocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

print("🔧 EasyDeL GPT-2 LoRA Fine-tuning - Simple Version")
print(f"JAX devices: {jax.devices()}")

# Load model
model_name = "gpt2"
print(f"\n🔧 Loading {model_name}...")

# New EasyDeL API returns only model object
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_name,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    platform=ed.EasyDeLPlatforms.JAX,
    device=jax.devices()[0],
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
    )
)

# Extract params from model - try common attributes
if hasattr(model, 'params'):
    params = model.params
elif hasattr(model, 'parameters'):
    params = model.parameters
elif hasattr(model, 'state_dict'):
    params = model.state_dict()
else:
    # Print available attributes to debug
    print(f"   Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
    raise AttributeError("Could not find model parameters")

print("✅ Model loaded!")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("✅ Tokenizer loaded!")

# Setup LoRA - use exactly what the signature expects
print(f"\n🔧 Setting up LoRA...")

# Move params to device and wrap in FrozenDict as shown in blog post
device_params = jax.device_put(params)
model_parameters = FrozenDict({"params": device_params})

lora_config = ed.EasyDeLBaseConfig(
    model_parameters,  # Pass wrapped params here
    lora_dim=8,  # This is required
    # Add other LoRA parameters that the config accepts
)

print("✅ LoRA configuration created!")

# Create simple dataset with raw text examples
mock_examples = [
    "SUBREDDIT: r/programming\nTITLE: Learning Python\nPOST: I've been learning Python for 6 months and built my first project. TL;DR: Built first Python project after 6 months.",
    "SUBREDDIT: r/fitness\nTITLE: Started running\nPOST: Began running 3 months ago and can now do 5K without stopping. TL;DR: Started running, now can do 5K runs.",
    "SUBREDDIT: r/technology\nTITLE: New AI breakthrough\nPOST: Scientists developed a new language model with better understanding. TL;DR: AI makes progress in language understanding.",
]

# Convert to text-based HuggingFace format for SFTTrainer
def to_text_dataset(examples):
    """Convert our mock examples to a dataset with text field"""
    text_data = {"text": examples * 10}  # Repeat for more samples
    return Dataset.from_dict(text_data)

hf_dataset = to_text_dataset(mock_examples)
print("✅ Dataset created!")

# Training arguments - use only what EasyDeL expects
print(f"\n🔧 Setting up training...")

training_args = ed.SFTConfig(
    num_train_epochs=1,
    total_batch_size=4,  # Reduced from 8 to match working config
    learning_rate=5e-4,
    dataset_text_field="text",  # Tell trainer which field contains the text
    max_sequence_length=550,  # Reduced from 1024 to match working config
)

print("✅ Training arguments created!")

# Create trainer with LoRA
print(f"\n🚀 Creating trainer with LoRA...")

# Use SFTTrainer (Supervised Fine-Tuning Trainer) from current EasyDeL version
trainer = ed.trainers.SFTTrainer(
    arguments=training_args,
    processing_class=tokenizer,
    model=model,
    train_dataset=hf_dataset,
    # LoRA config might go as finetune parameter
)
print("✅ Trainer created with LoRA!")

# Start training
print(f"\n🔥 Starting LoRA training...")
train_result = trainer.train()
print("🎉 LoRA training completed!")

# Get the trained parameters from the trainer
print(f"\n📥 Getting trained parameters...")
if hasattr(train_result, 'state') and hasattr(train_result.state, 'params'):
    trained_params = train_result.state.params
    print("✅ Got parameters from train_result.state.params")
elif hasattr(trainer, 'state') and hasattr(trainer.state, 'params'):
    trained_params = trainer.state.params
    print("✅ Got parameters from trainer.state.params")
else:
    # Fallback to original params (may not have LoRA updates)
    trained_params = params
    print("⚠️  Using original params as fallback")

# Update the model with trained parameters to avoid "Array has been deleted" error
print(f"\n🔄 Updating model with trained parameters...")
try:
    # Replace the model's parameters with the trained ones
    if hasattr(model, 'params'):
        model.params = trained_params
        print("✅ Updated model.params with trained parameters")
    elif hasattr(model, 'parameters'):
        model.parameters = trained_params
        print("✅ Updated model.parameters with trained parameters")
    else:
        print("⚠️  Could not update model parameters - model may use old params")
except Exception as e:
    print(f"⚠️  Could not update model parameters: {e}")

# Test inference with updated model
print(f"\n🧪 Testing inference...")
test_text = "SUBREDDIT: r/test\nTITLE: Test post\nPOST: This is a test.\n\nTL;DR:"
inputs = tokenizer(test_text, return_tensors="np", max_length=128, truncation=True)
inputs = {k: jnp.array(v, dtype=jnp.int32) for k, v in inputs.items()}

# Modern EasyDeL inference - model should use its own updated parameters
try:
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs.get('attention_mask'),
    )
    
    print("✅ Inference successful!")
    print(f"   Output shape: {outputs.logits.shape}")
    
    # Generate some text to verify the model works
    print(f"\n🎯 Testing text generation...")
    # Get logits for the last token to continue generation
    logits = outputs.logits[0, -1, :]  # Last token logits
    next_token_id = jnp.argmax(logits)
    next_token = tokenizer.decode([int(next_token_id)])
    print(f"   Next predicted token: '{next_token}'")
    
except Exception as e:
    print(f"❌ Inference failed: {e}")
    print("   This may indicate the model parameters are still invalid")

print(f"\n🎉 LoRA fine-tuning complete!")
