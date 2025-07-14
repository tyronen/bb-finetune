from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForSequenceClassificationWithValueHead, RewardTrainer
from datasets import load_dataset

# 1. Load tokenizer and SFT model (already merged)
tokenizer = AutoTokenizer.from_pretrained('path/to/your/sft-model')
sft_model = AutoModelForCausalLM.from_pretrained('path/to/your/sft-model')

# 2. Add value head
reward_model = AutoModelForSequenceClassificationWithValueHead.from_pretrained(sft_model)

# 3. Prepare dataset
dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="train")

def preprocess(batch):
    prompt = batch["prompt"]
    chosen = batch["chosen"]
    rejected = batch["rejected"]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

dataset = dataset.map(preprocess)

training_args = TrainingArguments(
    output_dir="./reward_model_output",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    fp16=True,
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    remove_unused_columns=False,
)

# 4. Train using RewardTrainer (from TRL)
trainer = RewardTrainer(
    model=reward_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()