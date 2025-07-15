import logging
from contextlib import nullcontext

import torch
from torch.amp import autocast, GradScaler

BASE = "Qwen/Qwen3-0.6B"
DATA_DIR = "data"
TMP_DIR = "/tmp"
SFT_DIR = "data/sft"
REWARD_DIR = "data/reward"
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
