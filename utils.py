import logging
from contextlib import nullcontext

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import GenerationConfig

BASE = "Qwen/Qwen3-0.6B"
DATA_DIR = "data"
TMP_DIR = "/dev/shm/.cache"
SFT_DIR = "data/sft"
REWARD_DIR = "data/reward"
PPO_DIR = "data/ppo"
max_input_length = 550


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def amp_components(device, train=False):
    if device.type == "cuda" and train:
        return autocast(device_type="cuda"), GradScaler()
    else:
        # fall-back: no automatic casting, dummy scaler
        return nullcontext(), GradScaler(enabled=False)


def evaluate_normalized_reward_score(
        model, reward_model, rw_tokenizer, dataset, tokenizer, num_samples
):
    """
    Generate responses from the model and normalize their reward by reference label score.
    """
    max_new_tokens = 100
    rewards = []
    device = get_device()
    reward_model.eval()

    for i, sample in tqdm(enumerate(dataset), total=min(num_samples, len(dataset))):
        if i >= num_samples:
            break

        input_text = sample["query"]
        reference_summary = sample["label"]

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Generate model output
        response_token_ids = model.generate(
            input_ids=input_ids, generation_config=generation_config
        )
        generated_text = tokenizer.decode(
            response_token_ids[0], skip_special_tokens=True
        )

        # Prepare full inputs
        full_gen_input = f"{input_text}\n{generated_text}"
        full_ref_input = f"{input_text}\n{reference_summary}"

        # Tokenize both
        gen_inputs = rw_tokenizer(
            full_gen_input, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        ref_inputs = rw_tokenizer(
            full_ref_input, return_tensors="pt", truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            gen_logits = reward_model(**gen_inputs).logits
            ref_logits = reward_model(**ref_inputs).logits

            gen_reward = torch.sigmoid(gen_logits.squeeze()).item()
            ref_reward = torch.sigmoid(ref_logits.squeeze()).item()

            if ref_reward == 0:
                normalized_reward = 0.0
            else:
                normalized_reward = (
                        gen_reward / ref_reward
                )  # or: gen_reward - ref_reward

        rewards.append(normalized_reward)

    return np.mean(rewards), np.std(rewards)
