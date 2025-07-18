import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import utils


def get_reward_score(prompt, summary):
    input_text = prompt.strip() + "\n" + summary.strip() + tokenizer.eos_token
    inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        # If your model outputs logits directly:
        score = outputs.logits.squeeze().item()
    return score


device = utils.get_device()
REWARD_MODEL_PATH = utils.REWARD_DIR

tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH, trust_remote_code=True
)

model.eval()
model.to(device)

dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="test[2000:2050]")
# dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="train[:20]")


results = []

for ex in tqdm(dataset, desc="Evaluating reward model"):
    prompt = ex["prompt"]
    chosen = ex["chosen"]
    rejected = ex["rejected"]

    chosen_score = get_reward_score(prompt, chosen)
    rejected_score = get_reward_score(prompt, rejected)
    preferred = "chosen" if chosen_score > rejected_score else "rejected"

    results.append(
        {
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "preferred": preferred,
            "correct": preferred == "chosen",  # Whether model agrees with human
            "prompt": prompt[:120],  # Truncate for printing
            "chosen": chosen[:100],
            "rejected": rejected[:100],
        }
    )

correct = sum(r["correct"] for r in results)
print(
    f"Model preferred 'chosen' {correct}/{len(results)} times ({100 * correct / len(results):.1f}% accuracy)"
)

for r in results[:3]:
    print("=" * 40)
    print("Prompt:", r["prompt"])
    print("Chosen:", r["chosen"])
    print("Rejected:", r["rejected"])
    print("Chosen score:", r["chosen_score"])
    print("Rejected score:", r["rejected_score"])
    print("Model preferred:", r["preferred"])
