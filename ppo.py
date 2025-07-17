import time

import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    GenerationConfig,
    AutoConfig,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import (
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)

import utils

scaling_factor = 200


def build_dataset(
        dataset_name, tokenizer, input_min_text_length, input_max_text_length
):
    """
    Preprocess the dataset and return train/valid/test splits with input_ids.

    Parameters:
    - model_name (str): Name or path of the tokenizer/model.
    - dataset_name (str): Name of the Hugging Face dataset.
    - input_min_text_length (int): Minimum character length of prompt.
    - input_max_text_length (int): Maximum character length of prompt.

    Returns:
    - dataset (datasets.DatasetDict): Tokenized dataset with train/valid/test splits.
    """

    # Load all splits
    dataset = load_dataset(dataset_name)

    def preprocess(split):
        # Filter by character length of prompt
        split = split.filter(
            lambda x: input_min_text_length < len(x["prompt"]) <= input_max_text_length,
            batched=False,
        )

        def tokenize(sample):
            prompt = f"{sample['prompt']}\n\n"
            inputs = tokenizer(prompt, truncation=True, max_length=1024)
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],  # good to have!
                "label": sample["label"],
            }

        split = split.map(tokenize, batched=False)
        split = split.remove_columns(
            [
                col
                for col in split.column_names
                if col not in ("input_ids", "attention_mask", "label")
            ]
        )
        split.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            output_all_columns=True,
        )

        return split

    dataset["train"] = preprocess(
        dataset["train"].select(range(117622 // scaling_factor))
    )
    dataset["valid"] = preprocess(
        dataset["valid"].select(range(6447 // scaling_factor))
    )
    dataset["test"] = preprocess(dataset["test"].select(range(6553 // scaling_factor)))
    for split in ("train", "valid"):
        dataset[split] = dataset[split].remove_columns(["label"])

    return dataset


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"\ntrainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


def main():
    wandb.init(entity="mlx-institute", project="ppo")

    tokenizer = AutoTokenizer.from_pretrained(utils.BASE, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = build_dataset(
        dataset_name="CarperAI/openai_summarize_tldr",
        tokenizer=tokenizer,
        input_min_text_length=200,
        input_max_text_length=1000,
    )
    ppo_model = AutoModelForCausalLM.from_pretrained(
        utils.SFT_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    ppo_model.config.pad_token_id = tokenizer.pad_token_id

    print(
        f"PPO model parameters to be updated (ValueHead + 769 params):\n{print_number_of_trainable_model_parameters(ppo_model)}\n"
    )

    ppo_model.config.eos_token_id = tokenizer.eos_token_id
    ppo_model.config.use_cache = False  # PPO doesn’t need past-key-values
    ppo_model.config.return_dict = True  # make forward return a ModelOutput

    # load the HF config you already have on disk
    base_cfg = AutoConfig.from_pretrained(utils.SFT_DIR)

    # turn that into a GenerationConfig (a dataclass)
    gen_cfg = GenerationConfig(**base_cfg.to_dict())

    # assign it to your PPO model
    ppo_model.generation_config = gen_cfg
    ref_model = create_reference_model(ppo_model)
    ref_model.config.use_cache = False
    ref_model.config.return_dict = True
    ref_model.generation_config = gen_cfg
    print(
        f"Reference model parameters to be updated:\n{print_number_of_trainable_model_parameters(ref_model)}\n"
    )

    rw_tokenizer = AutoTokenizer.from_pretrained(utils.REWARD_DIR)
    rw_model = AutoModelForSequenceClassification.from_pretrained(
        utils.REWARD_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    rw_model.config.return_dict = True
    print("Reward model created")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        utils.SFT_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        num_labels=1,
    )
    value_model.config.return_dict = True
    print("Value model created")
    learning_rate = 1.41e-5
    max_ppo_epochs = 1
    mini_batch_size = 4
    batch_size = 16

    device = utils.get_device()
    ref_model.to(device)
    rw_model.to(device)
    ppo_model.to(device)
    value_model.to(device)

    config = PPOConfig(
        learning_rate=learning_rate,
        num_ppo_epochs=max_ppo_epochs,
        mini_batch_size=mini_batch_size,
        batch_size=batch_size,
        report_to="wandb",
    )

    ppo_trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=ppo_model,
        ref_model=ref_model,
        reward_model=rw_model,
        value_model=value_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
    )
    print(
        f"GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB / {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB"
    )
    # this is from your PPOConfig:
    num_epochs = config.num_ppo_epochs  # e.g. 1
    B = config.batch_size  # total samples collected per epoch
    b = config.mini_batch_size  # samples per micro-batch
    G = config.gradient_accumulation_steps  # defaults to 1 if absent

    # total samples (trajectories) seen in one full PPO run:
    total_samples = num_epochs * B

    # total forward/backward passes (i.e. gradient-accumulation steps):
    #   first, how many micro-batches per epoch: B / b
    #   then how many optimizer steps per epoch: (B/b) / G
    optimizer_steps = num_epochs * (B // b) // G
    # Use the built‑in training loop
    start = time.time()
    ppo_trainer.train()
    elapsed = max(1.0, time.time() - start)

    samples_per_sec = total_samples / elapsed
    updates_per_sec = optimizer_steps / elapsed

    print(f"PPO elapsed time: {elapsed:.2f}s")
    print(f"  → samples/sec: {samples_per_sec:.1f}")
    print(f"  → updates/sec: {updates_per_sec:.1f}")
    wandb.log(
        {
            "train_time": elapsed,
            "throughput/samples_per_s": samples_per_sec,
            "throughput/updates_per_s": updates_per_sec,
        }
    )
    ppo_trainer.save_model(utils.PPO_DIR)
    ppo_trainer.generate_completions()
    mean, std = utils.evaluate_normalized_reward_score(
        ref_model, rw_model, rw_tokenizer, dataset["test"], tokenizer, num_samples=20
    )
    print(f"Ref model: mean: {mean:.2f} std: {std:.2f}")
    mean, std = utils.evaluate_normalized_reward_score(
        ppo_model, rw_model, rw_tokenizer, dataset["test"], tokenizer, num_samples=20
    )
    print(f"PPO model: mean: {mean:.2f} std: {std:.2f}")
    wandb.finish(0)


if __name__ == "__main__":
    main()
