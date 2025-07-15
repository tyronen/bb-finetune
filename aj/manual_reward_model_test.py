from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm


class ValueHead(nn.Module):
    """Simple linear value head for reward modeling."""
    def __init__(self, hidden_size):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # Expecting (batch_size, seq_len, hidden_size)
        # Take final token's hidden state for each example
        if hidden_states.dim() == 3:
            x = hidden_states[:, -1, :]
        else:
            x = hidden_states
        return self.value_head(x)  # shape: (batch, 1)ch, 1)


class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # store the LM backbone
        self.base_model = base_model
        # simple linear head on top of the last hidden state's final token
        self.value_head = ValueHead(base_model.config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        # single forward through the backbone
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state        # (batch, seq_len, hidden)
        pooled      = last_hidden[:, -1, :]            # (batch, hidden)
        rewards     = self.value_head(pooled).squeeze(-1)  # (batch,)
        return rewards


def get_reward_score(prompt, summary):
    input_text = prompt + summary
    inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        # If your model outputs logits directly:
        score = outputs.squeeze().item()
    return score


device = "cuda" if torch.cuda.is_available() else "cpu"

REWARD_MODEL_PATH = "qwen-reward-manual-checkpoint"
SFT_MODEL_PATH = "qwen-sft-checkpoint/merged"

tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
base_model = AutoModel.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)

model = RewardModel(base_model)
model.load_state_dict(torch.load(f"{REWARD_MODEL_PATH}/full_reward_model.pt", map_location=device))
model.to(device)
model.eval()

dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="valid1[2000:2050]")
#dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="train[:20]")


results = []

for ex in tqdm(dataset, desc="Evaluating reward model"):
    prompt = ex["prompt"]
    chosen = ex["chosen"]
    rejected = ex["rejected"]

    chosen_score = get_reward_score(prompt, chosen)
    rejected_score = get_reward_score(prompt, rejected)
    preferred = "chosen" if chosen_score > rejected_score else "rejected"

    results.append({
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "preferred": preferred,
        "correct": preferred == "chosen",  # Whether model agrees with human
        "prompt": prompt[:120],  # Truncate for printing
        "chosen": chosen[:100],
        "rejected": rejected[:100],
    })


correct = sum(r["correct"] for r in results)
print(f"Model preferred 'chosen' {correct}/{len(results)} times ({100*correct/len(results):.1f}% accuracy)")


for r in results[:3]:
    print("="*40)
    print("Prompt:", r["prompt"])
    print("Chosen:", r["chosen"])
    print("Rejected:", r["rejected"])
    print("Chosen score:", r["chosen_score"])
    print("Rejected score:", r["rejected_score"])
    print("Model preferred:", r["preferred"])
