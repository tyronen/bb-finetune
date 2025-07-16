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
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

import utils


def create_comparison_dataset(path, split):
    dataset = load_dataset(path, split=split)
    pairs = []
    for sample in dataset:
        if sample["chosen"] == sample["rejected"]:
            continue
        if len(sample["chosen"].split()) < 5 or len(sample["rejected"].split()) < 5:
            continue
        pairs.append(
            {
                "chosen": sample["prompt"] + "\n" + sample["chosen"],
                "rejected": sample["prompt"] + "\n" + sample["rejected"],
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


class QwenRewardModel(nn.Module):
    def __init__(self, sft):
        super().__init__()
        self.config = sft.config
        self.model = sft
        self.v_head = nn.Linear(self.config.hidden_size, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(utils.SFT_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        model_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )

        hidden_states = model_outputs.hidden_states[-1]

        rewards = self.v_head(hidden_states).squeeze(-1)

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2

        chosen_mask = attention_mask[:bs]
        rejected_mask = attention_mask[bs:]

        chosen_lengths = chosen_mask.sum(dim=1) - 1
        rejected_lengths = rejected_mask.sum(dim=1) - 1

        # Get end rewards efficiently (no loops!)
        batch_indices = torch.arange(bs, device=input_ids.device)
        chosen_end_scores = rewards[batch_indices, chosen_lengths]
        rejected_end_scores = rewards[batch_indices + bs, rejected_lengths]

        loss = -torch.nn.functional.logsigmoid(
            chosen_end_scores - rejected_end_scores
        ).mean()
        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }


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
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result


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
    training_args = TrainingArguments(
        bf16=True,
        dataloader_pin_memory=True,
        eval_steps=10000,
        eval_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="steps",
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        lr_scheduler_kwargs={"min_lr": 1e-6},
        lr_scheduler_type="cosine_with_min_lr",
        logging_steps=1000,
        max_grad_norm=1.0,
        num_train_epochs=1,
        output_dir=utils.REWARD_DIR,
        per_device_eval_batch_size=per_device_batch_size,
        per_device_train_batch_size=per_device_batch_size,
        remove_unused_columns=False,
        report_to="wandb",
        save_steps=0,
        save_strategy="no",
        warmup_steps=500,
        weight_decay=0.01,
    )

    # Initialize the reward model from the (supervised) fine-tuned Qwen
    model = AutoModelForCausalLM.from_pretrained(
        utils.SFT_DIR, torch_dtype=torch.bfloat16
    )
    lora_cfg = LoraConfig(
        r=32,  # rank of LoRA matrices
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen uses these
    )

    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model = QwenRewardModel(model)
    torch.nn.init.normal_(model.v_head.weight, std=0.01)
    model.to(device)

    # Make pairwise datasets for training
    train_dataset = cache_pairwise_dataset(
        tokenizer, "train_pairwise_dataset.pt", "train"
    )
    val_dataset = cache_pairwise_dataset(tokenizer, "val_pairwise_dataset.pt", "test")
    val_dataset = Subset(val_dataset, range(2000))
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
    # Merge LoRA into base
    merged_model = model.model.merge_and_unload()

    # (Optional) Save the reward head weights if needed
    torch.save(model.v_head.state_dict(), os.path.join(utils.REWARD_DIR, "v_head.pt"))

    # Save the merged base model and tokenizer
    merged_model.save_pretrained(utils.REWARD_DIR)
    tokenizer.save_pretrained(utils.REWARD_DIR)
    wandb.finish(0)


if __name__ == "__main__":
    main()
