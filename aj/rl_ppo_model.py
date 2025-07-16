import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, GenerationConfig, DataCollatorWithPadding, TrainerCallback
from transformers.modeling_outputs import CausalLMOutputWithPast
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from datasets import load_dataset
from tqdm import tqdm
import os
import inspect
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


# print(inspect.getfile(PPOTrainer))
# print("trl version:", __import__('trl').__version__)
# print("PPOTrainer:", PPOTrainer.__module__, PPOTrainer.__init__.__code__.co_varnames)

# import sys; sys.exit()

def force_return_dict_forward(self, *args, **kwargs):
    kwargs['return_dict'] = True
    output = self.__class__.forward(self, *args, **kwargs)
    if isinstance(output, tuple):
        return CausalLMOutputWithPast(
            logits=output[0],
            past_key_values=output[1] if len(output) > 1 else None,
        )
    return output


# ==== Reward function ====
def get_reward(prompt, summary):
    """
    Use reward_model to score a prompt+summary pair.
    Returns a scalar reward.
    """
    # Format input for reward model (copy your test script logic here if needed)
    input_text = prompt.strip() + "\n" + summary.strip()
    inputs = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        reward = reward_model(**inputs).logits.squeeze().item()
    return reward


# def evaluate_policy(model, tokenizer, reward_model, eval_prompts, max_new_tokens=64, device="cuda"):
#     model.eval()
#     reward_model.eval()
#     rewards = []
#     for prompt in tqdm(eval_prompts, desc="Evaluating policy"):
#         input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#         with torch.no_grad():
#             output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
#         generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         # Optionally, strip prompt from generated if model echoes prompt
#         gen_summary = generated[len(prompt):].strip()
#         reward = get_reward(prompt, gen_summary)
#         rewards.append(reward)
#     avg_reward = sum(rewards) / len(rewards)
#     print(f"Average eval reward: {avg_reward:.3f}")
#     return avg_reward


def _score(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    hidden_states: either
      - (batch, seq_len, hidden_size)  → return (batch, seq_len)
      - (batch, hidden_size)           → return (batch, 1)
    """
    # If they passed the whole sequence
    if hidden_states.dim() == 3:
        # v_head accepts (..., hidden_size) and outputs (..., 1)
        values = self.v_head(hidden_states)    # (batch, seq_len, 1)
        return values.squeeze(-1)              # (batch, seq_len)

    # If they passed only the last token's hidden state
    elif hidden_states.dim() == 2:
        # v_head on (batch, hidden_size) → (batch, 1)
        values = self.v_head(hidden_states)    # (batch, 1)
        # **Do not** squeeze that last dim: keep it as (batch, 1)
        return values                          # (batch, 1)

    else:
        raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Paths ====
SFT_MODEL_PATH = "qwen-sft-checkpoint/merged"
TOKENIZER_PATH = "qwen-sft-checkpoint/checkpoint-3000"
REWARD_MODEL_PATH = "qwen-reward-checkpoint"
SAVE_PATH = "qwen-ppo-policy-checkpoint"
os.makedirs(SAVE_PATH, exist_ok=True)

# ==== Tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorWithPadding(tokenizer)

# ==== Policy Model ====
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(SFT_MODEL_PATH)
policy_model.forward = force_return_dict_forward.__get__(policy_model, policy_model.__class__)

# ==== Reference Model ====
ref_model = create_reference_model(policy_model)
ref_model.to(device)

value_model = policy_model
#value_model.score = _score.__get__(value_model, value_model.__class__)
policy_model.to(device)
value_model.to(device)

for m in [policy_model, ref_model, value_model]:
    if m is not None and not hasattr(m, "score"):
        m.score = _score.__get__(m, m.__class__)
    # If the model has a .model or .pretrained_model attribute, patch that too
    if hasattr(m, "model") and not hasattr(m.model, "score"):
        m.model.score = _score.__get__(m.model, m.model.__class__)
    if hasattr(m, "pretrained_model") and not hasattr(m.pretrained_model, "score"):
        m.pretrained_model.score = _score.__get__(m.pretrained_model, m.pretrained_model.__class__)


if not hasattr(policy_model, "generation_config"):
    policy_model.generation_config = GenerationConfig.from_pretrained(SFT_MODEL_PATH)

policy_model.generation_config.return_dict = True
policy_model.generation_config.pad_token_id = tokenizer.pad_token_id
policy_model.generation_config.eos_token_id = tokenizer.eos_token_id
policy_model.generation_config.bos_token_id = tokenizer.bos_token_id

if hasattr(policy_model, "pretrained_model"):
    policy_model.model = policy_model.pretrained_model
    policy_model.base_model_prefix = "model"

# ==== Reward Model ====
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH,
    num_labels=1,
    trust_remote_code=True,
)
reward_model.config.pad_token_id = tokenizer.pad_token_id
reward_model.eval()
reward_model.to(device)

# ==== Dataset ====
train_dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")
#prompts = [item["prompt"] for item in dataset]
train_prompts = [item["prompt"] for item in train_dataset]  # use only 100 for quick test

tokenized_prompts = tokenizer(
    train_prompts,
    padding=False,  # No padding here, let collator do it
    truncation=True,
    max_length=512,
    return_tensors=None,      # Output will be list of dicts, not batch tensor
)
# tokenizer returns a dict of lists, so convert to list of dicts:
prompts = [
    {"input_ids": input_ids, "index": i}
    for i, input_ids in enumerate(tokenized_prompts["input_ids"])
]

N_EVAL = 2000

eval_dataset = load_dataset("CarperAI/openai_summarize_tldr", split="valid")
eval_prompts = [item["prompt"] for item in eval_dataset][:N_EVAL]  # e.g., first 100 for quick eval
eval_summaries = [item["label"] for item in eval_dataset][:N_EVAL] 

tokenized_eval_prompts = tokenizer(
    eval_prompts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors=None,   # so we get Python lists
)

eval_dataset_ppo = [
    {"input_ids": input_ids, "index": i}
    for i, input_ids in enumerate(tokenized_eval_prompts["input_ids"])
]


# ==== PPO Config ====
ppo_config = PPOConfig(
    output_dir=SAVE_PATH,                # Where to save checkpoints
    per_device_train_batch_size=2,          # Used for DataLoader, not PPO batch size
    per_device_eval_batch_size=2,           # Same as above
    gradient_accumulation_steps=1,
    learning_rate=1.41e-5,
    fp16=True,
    bf16=False,
    # RLHF/PPO settings:
    batch_size=2,              # PPO batch size (outer loop)
    mini_batch_size=1,          # PPO mini batch (for advantage estimation)
    num_ppo_epochs=1,
    whiten_rewards=False,       # Most RLHF doesn't whiten
    kl_coef=0.01,               # Initial KL penalty coefficient
    cliprange=0.2,
    vf_coef=0.1,
    cliprange_value=0.2,
    gamma=1.0,
    lam=0.95,
    temperature=1.2,            # Or 0.7, as you like
    exp_name="ppo_config",
    eval_strategy="steps",
    eval_steps=500,           # e.g. every 500 training steps
    eval_on_start=False,
    max_steps=9000,
    save_safetensors=False,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    logging_strategy="steps",     # log according to step counts
    logging_steps=500,
    disable_tqdm=False,
)

ppo_trainer = PPOTrainer(
    ppo_config,      # PPOConfig object with all settings
    tokenizer,
    policy_model,    # Your policy model (with ValueHead for PPO)
    ref_model,       # Reference model (frozen SFT, KL penalty)
    reward_model,    # Reward model (SequenceClassification)
    prompts,         # List of prompt strings for RL
    value_model,            # value_model (usually None for LM RLHF)
    data_collator,            # data_collator (optional)
    eval_dataset_ppo,            # eval_dataset (optional)
)

# generation_kwargs = {
#     "min_length": -1,
#     "top_k": 0.0,
#     "top_p": 1.0,
#     "do_sample": True,
#     "pad_token_id": tokenizer.eos_token_id,
#     "max_new_tokens": 20,
# }

ppo_trainer.train()

ppo_trainer.save_model(SAVE_PATH)

final_model = os.path.join(SAVE_PATH, "ppo_policy_final")
print(f"Saved final PPO model to {final_model}")