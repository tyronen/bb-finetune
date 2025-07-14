import random

import evaluate
import numpy as np
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator,
)

import utils

BASE = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "data"
max_input_length = 550


class TLDRDataset(Dataset):
    def __init__(self, tokenizer, split, max_length):
        dataset = load_dataset("CarperAI/openai_summarize_tldr", split=split)
        self.examples = [sample["prompt"] + sample["label"] for sample in dataset]
        self.examples = self.examples[:2000] if "valid" in split else self.examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.examples[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(enc["input_ids"]),  # teacher forcing
        }


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def main():
    device = utils.get_device()
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        attn_implementation="flash_attention_2",  # ❶ fast kernels
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    lora_cfg = LoraConfig(
        r=4,  # rank of LoRA matrices
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen uses these
    )

    model = get_peft_model(model, lora_cfg)
    model.to(device)
    model.resize_token_embeddings(len(tokenizer))  # adjust token count
    model.config.pad_token_id = tokenizer.eos_token_id

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds = tokenizer.batch_decode(eval_preds.predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(eval_preds.label_ids, skip_special_tokens=True)
        return rouge.compute(predictions=preds, references=labels)

    training_args = Seq2SeqTrainingArguments(
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=True,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=500,
        gradient_accumulation_steps=1,
        learning_rate=3e-5,
        load_best_model_at_end=True,
        log_level="info",
        logging_steps=50,
        lr_scheduler_type="cosine",
        max_steps=4000,
        optim="adamw_torch_fused",  # fused optimiser
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=16,
        per_device_train_batch_size=8,
        predict_with_generate=True,
        report_to="wandb",
        save_steps=1000,
        save_strategy="steps",
        warmup_steps=100,
    )

    train_dataset = TLDRDataset(
        tokenizer,
        "train",
        max_length=max_input_length,
    )
    dev_dataset = TLDRDataset(
        tokenizer,
        "valid",
        max_length=max_input_length,
    )

    wandb.init(entity="mlx-institute", project="sft")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,  # stop after 3 idle epochs
                early_stopping_threshold=0.001,  # needs ≤0.1 % val-loss improvement
            )
        ],
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer._gen_kwargs = {"max_new_tokens": 128}
    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()
