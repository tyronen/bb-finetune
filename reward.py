import os

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
    def __init__(self, pairs, tokenizer, max_length):
        super().__init__()
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
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
        chosen_end_scores = []
        rejected_end_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        loss = 0
        for i in range(bs):
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])

            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(
                torch.sigmoid(c_truncated_reward - r_truncated_reward)
            ).mean()
        loss = loss / bs

        chosen_end_scores = torch.stack(chosen_end_scores)
        rejected_end_scores = torch.stack(rejected_end_scores)

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
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
    if os.path.exists(cache_path):
        print(f"Loading cached PairwiseDataset from {cache_path}. Ignore warning")
        return torch.load(cache_path)
    else:
        print("Building PairwiseDataset and caching...")
        pairs = create_comparison_dataset(
            "CarperAI/openai_summarize_comparisons", split
        )
        dataset = PairwiseDataset(pairs, tokenizer, max_length=utils.max_input_length)
        torch.save(dataset, cache_path)
        return dataset


def main():
    device = utils.get_device()
    wandb.init(entity="mlx-institute", project="reward")

    tokenizer = AutoTokenizer.from_pretrained(utils.SFT_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists(utils.REWARD_DIR):
        os.mkdir(utils.REWARD_DIR)

    training_args = TrainingArguments(
        bf16=True,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        eval_accumulation_steps=1,
        eval_steps=100,
        eval_strategy="steps",
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=10,
        num_train_epochs=1,
        output_dir=utils.REWARD_DIR,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
        report_to="wandb",
        save_steps=500,
        save_strategy="steps",
        save_total_limit=1,
        warmup_steps=100,
    )

    # Initialize the reward model from the (supervised) fine-tuned Qwen

    model = AutoModelForCausalLM.from_pretrained(utils.SFT_DIR)
    lora_cfg = LoraConfig(
        r=8,  # rank of LoRA matrices
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen uses these
    )

    model = get_peft_model(model, lora_cfg)
    model = QwenRewardModel(model)
    model.to(device)

    # Make pairwise datasets for training
    train_dataset = cache_pairwise_dataset(
        tokenizer, "train_pairwise_dataset.pt", "train"
    )
    val_dataset = cache_pairwise_dataset(tokenizer, "val_pairwise_dataset.pt", "test")
    val_dataset = Subset(val_dataset, range(500))
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
    trainer.train()
    model = model.merge_and_unload()
    model.save_pretrained(utils.REWARD_DIR)
    tokenizer.save_pretrained(utils.REWARD_DIR)
    wandb.finish(0)


if __name__ == "__main__":
    main()
