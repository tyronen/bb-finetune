import os
import shutil
import time

import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import utils

scaling_factor = 100


def create_comparison_dataset(path, split):
    dataset = load_dataset(path, split=split)
    pairs = []
    for sample in dataset:
        if sample["chosen"] == sample["rejected"]:
            continue
        if len(sample["chosen"].split()) < 5 or len(sample["rejected"].split()) < 5:
            continue
        prompt = sample["prompt"].split("POST:")[-1].strip()
        pairs.append(
            {
                "chosen": prompt + "\n" + sample["chosen"],
                "rejected": prompt + "\n" + sample["rejected"],
            }
        )
    return pairs


class PairwiseDataset(Dataset):
    @classmethod
    def from_tensors(cls, tensors):
        obj = cls.__new__(cls)
        obj.chosen_input_ids = tensors["chosen_input_ids"]
        obj.chosen_attn_masks = tensors["chosen_attn_masks"]
        obj.rejected_input_ids = tensors["rejected_input_ids"]
        obj.rejected_attn_masks = tensors["rejected_attn_masks"]
        return obj

    def __init__(self, pairs, tokenizer, max_length):
        super().__init__()
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            if not torch.all(
                    torch.eq(
                        chosen_encodings_dict["input_ids"],
                        rejected_encodings_dict["input_ids"],
                    )
            ).item():
                self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
                self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
                self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
                self.rejected_attn_masks.append(
                    rejected_encodings_dict["attention_mask"]
                )
        self.chosen_input_ids = torch.cat(self.chosen_input_ids, dim=0)  # [N, seq_len]
        self.chosen_attn_masks = torch.cat(self.chosen_attn_masks, dim=0)
        self.rejected_input_ids = torch.cat(self.rejected_input_ids, dim=0)
        self.rejected_attn_masks = torch.cat(self.rejected_attn_masks, dim=0)

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )


class QwenRewardModel(PreTrainedModel):
    def __init__(self, sft):
        super().__init__(sft.config)
        self.transformer = sft
        self.score = nn.Linear(self.config.hidden_size, 1, bias=False)
        self.config.problem_type = "regression"
        self.config.num_labels = 1

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            **kwargs,
    ):
        model_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = model_outputs.hidden_states[-1]

        rewards = self.score(hidden_states).squeeze(-1)

        # Split the inputs and rewards into two parts, chosen and rejected
        end_mask = attention_mask.sum(dim=1) - 1
        logits = rewards[torch.arange(rewards.size(0)), end_mask].unsqueeze(-1)
        loss = None
        if labels is not None:
            chosen_end_scores, rejected_end_scores = logits.chunk(2, dim=0)
            loss = -torch.nn.functional.logsigmoid(
                chosen_end_scores - rejected_end_scores
            ).mean()

        return SequenceClassifierOutput(loss=loss, logits=logits)


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.stack([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.stack(
            [f[1] for f in data] + [f[3] for f in data]
        )
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch


def compute_metrics(eval_preds):
    preds = eval_preds.predictions.squeeze()  # (N,)
    half = preds.shape[0] // 2
    chosen_end_scores = preds[:half]
    rejected_end_scores = preds[half:]
    acc = (chosen_end_scores > rejected_end_scores).mean().item()
    return {"accuracy": acc}


def cache_pairwise_dataset(tokenizer, cache_path, split):
    tmp_path = f"{utils.TMP_DIR}/{cache_path}"
    data_path = f"{utils.DATA_DIR}/{cache_path}"

    # 1. If the file exists in /tmp, load from there
    if os.path.exists(tmp_path):
        print(f"Loading cached PairwiseDataset from {tmp_path}")
        tensors = torch.load(tmp_path)
        return PairwiseDataset.from_tensors(tensors)

    # 2. If it exists in data/, copy to /tmp, then load
    if os.path.exists(data_path):
        print(f"Copying {data_path} to {tmp_path}")
        shutil.copyfile(data_path, tmp_path)
        tensors = torch.load(tmp_path)
        return PairwiseDataset.from_tensors(tensors)

    # 3. Else, build and save to both data/ and /tmp
    print("Building PairwiseDataset and caching...")
    pairs = create_comparison_dataset("CarperAI/openai_summarize_comparisons", split)
    dataset = PairwiseDataset(pairs, tokenizer, max_length=utils.max_input_length)
    tensor_dict = {
        "chosen_input_ids": dataset.chosen_input_ids,
        "chosen_attn_masks": dataset.chosen_attn_masks,
        "rejected_input_ids": dataset.rejected_input_ids,
        "rejected_attn_masks": dataset.rejected_attn_masks,
    }
    torch.save(tensor_dict, data_path)
    torch.save(tensor_dict, tmp_path)
    return dataset


def main():
    device = utils.get_device()
    wandb.init(entity="mlx-institute", project="reward")

    tokenizer = AutoTokenizer.from_pretrained(utils.SFT_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists(utils.REWARD_DIR):
        os.mkdir(utils.REWARD_DIR)

    per_device_batch_size = 4
    gradient_accumulation_steps = 4
    all_possible_steps = 92534
    steps_per_epoch = (
            all_possible_steps
            // scaling_factor
            // per_device_batch_size
            // gradient_accumulation_steps
    )
    eval_steps = steps_per_epoch // 8
    logging_steps = eval_steps // 2
    warmup_steps = min(500, 2 * eval_steps)
    training_args = TrainingArguments(
        bf16=True,
        dataloader_pin_memory=True,
        eval_accumulation_steps=gradient_accumulation_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=1e-4,
        lr_scheduler_kwargs={"min_lr": 5e-6},
        lr_scheduler_type="cosine_with_min_lr",
        logging_steps=logging_steps,
        max_grad_norm=1.0,
        max_steps=steps_per_epoch,
        num_train_epochs=1,
        output_dir=utils.REWARD_DIR,
        per_device_eval_batch_size=per_device_batch_size,
        per_device_train_batch_size=per_device_batch_size,
        remove_unused_columns=False,
        report_to="wandb",
        save_steps=0,
        save_strategy="no",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
    )

    # Initialize the reward model from the (supervised) fine-tuned Qwen
    model = AutoModelForCausalLM.from_pretrained(
        utils.SFT_DIR, torch_dtype=torch.bfloat16
    )
    lora_cfg = LoraConfig(
        r=16,  # rank of LoRA matrices
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen uses these
    )

    model = get_peft_model(model, lora_cfg)
    model = QwenRewardModel(model)
    torch.nn.init.normal_(model.score.weight, std=0.01)
    model.to(device)

    # Make pairwise datasets for training
    train_dataset = cache_pairwise_dataset(
        tokenizer, "train_pairwise_dataset.pt", "train"
    )
    val_dataset = cache_pairwise_dataset(tokenizer, "val_pairwise_dataset.pt", "test")
    val_dataset = Subset(val_dataset, range(3000 // scaling_factor))
    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    start = time.time()
    trainer.train()
    end = time.time()
    print(f"WALL CLOCK training time: {end - start:.2f} seconds")
    merged = model.transformer.merge_and_unload()  # fuse LoRA into base
    config = merged.config
    config.problem_type = "regression"
    config.num_labels = 1
    merged.save_pretrained(f"{utils.REWARD_DIR}/merged")
    sc_model = AutoModelForSequenceClassification.from_pretrained(
        f"{utils.REWARD_DIR}/merged",
        trust_remote_code=True,
        config=config,
        ignore_mismatched_sizes=True,
    )
    sc_model.score.weight.data = model.score.weight.data.clone()
    sc_model.save_pretrained(utils.REWARD_DIR)
    tokenizer.save_pretrained(utils.REWARD_DIR)
    wandb.finish(0)


if __name__ == "__main__":
    main()
