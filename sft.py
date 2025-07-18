import time

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

import utils

scaling_factor = 20


class CustomTrainer(SFTTrainer):
    def get_eval_dataloader(self, eval_dataset=None):
        # Use only 2 workers for evaluation, for example
        dataloader = super().get_eval_dataloader(eval_dataset)
        dataloader.num_workers = 2
        return dataloader


def main():
    device = utils.get_device()
    wandb.init(entity="mlx-institute", project="sft")

    tokenizer = AutoTokenizer.from_pretrained(utils.BASE)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        utils.BASE,
        attn_implementation="flash_attention_2",  # ❶ fast kernels
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    lora_cfg = LoraConfig(
        r=8,  # rank of LoRA matrices
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen uses these
    )

    model = get_peft_model(model, lora_cfg)
    model.to(device)
    model.resize_token_embeddings(len(tokenizer))  # adjust token count
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False

    per_device_train_batch_size = 4
    gradient_accumulation_steps = 2
    all_possible_steps = 116722
    steps_per_epoch = (
            all_possible_steps
            // scaling_factor
            // per_device_train_batch_size
            // gradient_accumulation_steps
    )
    eval_steps = steps_per_epoch // 8
    logging_steps = eval_steps // 2
    warmup_steps = max(200, eval_steps)
    training_args = SFTConfig(
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=True,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=eval_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        load_best_model_at_end=True,
        log_level="info",
        logging_steps=logging_steps,
        lr_scheduler_type="constant_with_warmup",
        max_steps=steps_per_epoch,
        optim="adamw_torch_fused",  # fused optimiser
        output_dir=utils.SFT_DIR,
        per_device_eval_batch_size=4 * per_device_train_batch_size,
        per_device_train_batch_size=per_device_train_batch_size,
        report_to="wandb",
        save_steps=eval_steps,
        save_strategy="steps",
        warmup_ratio=0.05,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
    )

    train_dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")
    train_dataset = train_dataset.select(range(len(train_dataset) // scaling_factor))
    dev_dataset = load_dataset("CarperAI/openai_summarize_tldr", split="valid")
    dev_dataset = dev_dataset.select(range(len(dev_dataset) // scaling_factor))

    def build_chat(sample):
        return {
            "prompt": (
                "### TASK: Write a TL;DR summary for this Reddit post:\n\n"
                f"{sample['prompt'].split('POST:')[-1].strip()}\n\n"
                f"TL;DR:"
            ),
            "target": sample["label"],
        }

    def tokenize_chat(sample):
        full_text = sample["prompt"] + " " + sample["target"]
        prompt_ids = tokenizer(sample["prompt"], add_special_tokens=False)["input_ids"]
        full_enc = tokenizer(
            full_text,
            truncation=True,
            max_length=utils.max_input_length,
            padding="max_length",
        )
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        labels = input_ids.copy()
        labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_dataset = train_dataset.map(build_chat).map(
        tokenize_chat,
        remove_columns=train_dataset.column_names,
    )
    dev_dataset = dev_dataset.map(build_chat).map(
        tokenize_chat,
        remove_columns=dev_dataset.column_names,
    )

    # --- SFTTrainer ----------------------------------------------------
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        peft_config=lora_cfg,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,  # stop after 3 idle epochs
                early_stopping_threshold=0.001,  # needs ≤0.1 % val-loss improvement
            )
        ],
    )
    start = time.time()
    trainer.train()
    end = time.time()
    print(f"WALL CLOCK training time: {end - start:.2f} seconds")
    model = model.merge_and_unload()
    model.save_pretrained(utils.SFT_DIR)
    tokenizer.save_pretrained(utils.SFT_DIR)
    wandb.finish(0)


if __name__ == "__main__":
    main()
