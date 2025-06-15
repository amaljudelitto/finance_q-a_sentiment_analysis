import os
import json
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from utils import preprocess_data, compute_metrics

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Load datasets
qa_data = pd.read_csv("data/financial_qa_dataset.csv")
sentiment_data = pd.read_csv("data/sentiment_dataset.csv")

# Add task type
qa_data["task"] = "qa"
sentiment_data["task"] = "sentiment"

# Combine datasets
combined_df = pd.concat([qa_data, sentiment_data], ignore_index=True)
combined_df = preprocess_data(combined_df)

# Split dataset
train_texts, val_texts = train_test_split(combined_df, test_size=0.1, random_state=42)
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_texts),
    "validation": Dataset.from_pandas(val_texts)
})

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "NousResearch/TinyLLaMA-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Training args
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="steps",
    eval_steps=config["eval_steps"],
    logging_steps=config["logging_steps"],
    save_steps=config["save_steps"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    num_train_epochs=config["num_train_epochs"],
    learning_rate=config["learning_rate"],
    weight_decay=config["weight_decay"],
    save_total_limit=config["save_total_limit"],
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# Train
trainer.train()
model.save_pretrained("model")
tokenizer.save_pretrained("model")

