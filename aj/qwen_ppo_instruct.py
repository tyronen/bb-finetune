import shutil
import os
import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    get_peft_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


# ========= USER CONFIG ===========

sft_model_path = "./qwen-sft-instruct-checkpoint/merged"
reward_model_path = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
output_dir = "./qwen-ppo-rlhf-checkpoint"
dataset_name = "OpenAssistant/oasst1"
dataset_split = "validation"
prompt_column = "text"
eval_samples = 100  # number of eval prompts from the end

# PPO/Trainer params
per_device_train_batch_size = 1
gradient_accumulation_steps = 1
learning_rate = 3e-6
num_ppo_epochs = 1
total_episodes = 10000

# ================================

# -- ModelConfig --
model_args = ModelConfig(
    model_name_or_path=sft_model_path,
    trust_remote_code=True,
    torch_dtype="auto",
)

# -- PPOConfig --
ppo_args = PPOConfig(
    output_dir=output_dir,
    resume_from_checkpoint="/qwen-ppo-rlhf-checkpoint/checkpoint-1000",
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    total_episodes=total_episodes,
    num_ppo_epochs=num_ppo_epochs,
    logging_steps=100,
    fp16=True,
    bf16=False,
    batch_size=1,              # PPO batch size (outer loop)
    mini_batch_size=1,          # PPO mini batch (for advantage estimation)
    whiten_rewards=False,       # Most RLHF doesn't whiten
    kl_coef=0.01,               # Initial KL penalty coefficient
    cliprange=0.2,
    vf_coef=0.1,
    cliprange_value=0.2,
    gamma=1.0,
    lam=0.95,
    temperature=1.0,            # Or 0.7, as you like
    exp_name="ppo_config",
    eval_strategy="steps",
    eval_steps=1000,           # e.g. every 500 training steps
    save_strategy="steps",   # or "epoch" if you want to save every epoch
    save_steps=250,          # save every 250 steps, adjust as desired
    save_total_limit=3,      # keep last 3 checkpoints (optional)
)

# ========== MODEL AND TOKENIZER LOADING =========

tokenizer = AutoTokenizer.from_pretrained(
    sft_model_path, padding_side="left", trust_remote_code=True
)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

# Policy model (Qwen3 SFT)
policy = AutoModelForCausalLM.from_pretrained(sft_model_path, trust_remote_code=True)

# Reference policy
peft_config = get_peft_config(model_args)
ref_policy = None
if peft_config is None:
    ref_policy = AutoModelForCausalLM.from_pretrained(sft_model_path, trust_remote_code=True)

# Reward/Value models (deberta)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path, trust_remote_code=True, num_labels=1
)

value_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path, trust_remote_code=True, num_labels=1
)

# ========== DATA LOADING ==========

dataset = load_dataset(dataset_name, split=dataset_split)
train_dataset = dataset.select(range(len(dataset) - eval_samples))
eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

def prepare_dataset(dataset, tokenizer):
    def tokenize(element):
        outputs = tokenizer(
            element[prompt_column],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}
    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1,  # single process is safest, adjust as desired
    )

with PartialState().local_main_process_first():
    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)

# ========== PPO TRAINING ==========

trainer = PPOTrainer(
    args=ppo_args,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_policy,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)

trainer.train()

trainer.save_model(output_dir)
trainer.generate_completions()

print("PPO RLHF Qwen3 training complete! Model saved to:", output_dir)