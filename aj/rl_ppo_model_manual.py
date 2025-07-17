import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GenerationConfig
)
from datasets import load_dataset
from tqdm import tqdm


# === Define PPO policy+value wrapper ===
class PolicyWithValueHead(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        hidden_size = base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, **gen_kwargs):
        # Forward through LM, get logits and hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **gen_kwargs
        )
        # Last token hidden state -> value
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # (batch, hidden)
        values = self.value_head(last_hidden).squeeze(-1)  # (batch,)
        return outputs.logits, values


# ==== Reward function ====
def get_reward(prompts, responses):
    """
    Use reward_model to score a batch of prompt+summary pairs.
    Returns a tensor of rewards.
    """
    assert isinstance(prompts, list) and isinstance(responses, list)
    input_texts = [p.strip() + "\n" + r.strip() for p, r in zip(prompts, responses)]
    inputs = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(reward_model.device)

    with torch.no_grad():
        rewards = reward_model(**inputs).logits.squeeze(-1)
    return rewards  # shape: (batch,)


def evaluate_policy(policy_model, eval_prompts):
    policy_model.eval()
    rewards = []
    examples = []
    for prompt in tqdm(eval_prompts, desc="Eval prompts"):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            out = policy_model.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id
            )
        gen = tokenizer.decode(out[0], skip_special_tokens=True)
        summary = gen[len(prompt):].strip()
        rew = get_reward([prompt], [summary]).item()
        rewards.append(rew)
        if len(examples) < 1:
            examples.append((prompt, summary))
    avg_reward = sum(rewards) / len(rewards)
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Prompt: {examples[0][0]}\n\nSummary: {examples[0][1]}\n")
    policy_model.train()
    return avg_reward


def collate_fn(batch):
    # batch: list of strings (prompts)
    enc = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    return enc['input_ids'], enc['attention_mask']

# def force_return_dict_forward(self, *args, **kwargs):
#     kwargs['return_dict'] = True
#     output = self.__class__.forward(self, *args, **kwargs)
#     if isinstance(output, tuple):
#         return CausalLMOutputWithPast(
#             logits=output[0],
#             past_key_values=output[1] if len(output) > 1 else None,
#         )
#     return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Paths ====
SFT_MODEL_PATH = "qwen-sft-checkpoint/merged"
TOKENIZER_PATH = "qwen-sft-checkpoint/checkpoint-3000"
REWARD_MODEL_PATH = "qwen-reward-checkpoint"
SAVE_PATH = "qwen-ppo-policy-manual-checkpoint"
os.makedirs(SAVE_PATH, exist_ok=True)

# === Hyperparameters ===
learning_rate = 1.41e-5
batch_size = 2           # number of prompts per PPO batch
mini_batch_size = 1      # for advantage estimation
ppo_epochs = 4           # number of PPO epochs over each batch
gamma = 1.0              # reward discount (unused if no discounting)
kl_coef = 0.02           # KL penalty coefficient
value_coef = 0.5
max_new_tokens = 64
eval_interval = 100     # batches between evaluations
num_eval_prompts = 50    # number of eval prompts

# ==== Tokenizer ====
print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# ==== Policy Model ====
model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH)
policy_model = PolicyWithValueHead(model)
policy_model.to(device)

# ==== Reference Model ====
ref_model = policy_model
ref_model.to(device)

# ==== Value Model ====
value_model = policy_model
#value_model.score = _score.__get__(value_model, value_model.__class__)
value_model.to(device)

# for m in [policy_model, ref_model, value_model]:
#     if m is not None and not hasattr(m, "score"):
#         m.score = _score.__get__(m, m.__class__)
#     # If the model has a .model or .pretrained_model attribute, patch that too
#     if hasattr(m, "model") and not hasattr(m.model, "score"):
#         m.model.score = _score.__get__(m.model, m.model.__class__)
#     if hasattr(m, "pretrained_model") and not hasattr(m.pretrained_model, "score"):
#         m.pretrained_model.score = _score.__get__(m.pretrained_model, m.pretrained_model.__class__)


# if not hasattr(policy_model, "generation_config"):
#     policy_model.generation_config = GenerationConfig.from_pretrained(SFT_MODEL_PATH)

# policy_model.generation_config.return_dict = True
# policy_model.generation_config.pad_token_id = tokenizer.pad_token_id
# policy_model.generation_config.eos_token_id = tokenizer.eos_token_id
# policy_model.generation_config.bos_token_id = tokenizer.bos_token_id

# if hasattr(policy_model, "pretrained_model"):
#     policy_model.model = policy_model.pretrained_model
#     policy_model.base_model_prefix = "model"

# ==== Reward Model ====
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH,
    num_labels=1,
    trust_remote_code=True,
)
reward_model.config.pad_token_id = tokenizer.pad_token_id
reward_model.to(device)
reward_model.eval()

# ==== Datasets ====
train_data = load_dataset("CarperAI/openai_summarize_tldr", split="train")
#prompts = [item["prompt"] for item in dataset]
train_dataset = [x['prompt'] for x in train_data]
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)

eval_data = load_dataset("CarperAI/openai_summarize_tldr", split="valid")
eval_prompts = [x['prompt'] for x in eval_data][:num_eval_prompts]

# === PPO training loop ===
optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
MAX_STEPS = 9000

if __name__ == '__main__':
    for step, (input_ids, attention_mask) in enumerate(tqdm(train_dataloader, desc="PPO Training", unit="batch"), 1):
        if step > MAX_STEPS:
            print(f"Reached max steps: {MAX_STEPS}. Stopping training.")
            break

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # ---- Generate ----
        gen_out = policy_model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=0,
            top_p=1.0,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        seqs = gen_out.sequences
        prompt_lens = attention_mask.sum(dim=1)  # Each prompt's actual length

        responses = [
            tokenizer.decode(seqs[i, prompt_lens[i]:], skip_special_tokens=True)
            for i in range(seqs.size(0))
        ]

        # ---- Old logprobs & values ----
        old_logprobs = []
        for idx, logits in enumerate(gen_out.scores):
            tokens = seqs[:, prompt_lens.max() + idx]  # use max prompt len for all
            logp = torch.log_softmax(logits, dim=-1)
            old_logprobs.append(logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1))
        old_logprobs = torch.stack(old_logprobs, dim=1).sum(1)
        _, old_values = policy_model(input_ids=seqs)
        old_values = old_values.detach()

        # ---- Reward & advantages ----
        prompts = [tokenizer.decode(input_ids[i, :prompt_lens[i]], skip_special_tokens=True) for i in range(input_ids.size(0))]
        rewards = get_reward(prompts, responses).detach()
        advantages = rewards - old_values
        returns = rewards

        # ---- PPO updates ----
        for _ in range(ppo_epochs):
            logits, values = policy_model(input_ids=seqs)
            new_logprobs = []
            for idx, logit in enumerate(logits[:, prompt_lens.max():, :].unbind(1)):
                tokens = seqs[:, prompt_lens.max() + idx]
                logp = torch.log_softmax(logit, dim=-1)
                new_logprobs.append(logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1))
            new_logprobs = torch.stack(new_logprobs, dim=1).sum(1)

            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - values).pow(2).mean()
            kl = (old_logprobs - new_logprobs).mean()
            loss = policy_loss + value_coef * value_loss + kl_coef * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---- Checkpoint & Eval ----
        if step % eval_interval == 0:
            evaluate_policy(policy_model, eval_prompts)
            policy_model.model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)

    print("PPO training finished. Final model saved.")