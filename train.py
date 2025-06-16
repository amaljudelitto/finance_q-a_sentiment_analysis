from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import pandas as pd
import os

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load dataset
def load_data(path):
    df = pd.read_csv(path)
    dataset = df.rename(columns={"question": "input_text", "answer": "target_text"})
    dataset = dataset.dropna()

    from datasets import Dataset
    return Dataset.from_pandas(dataset)

dataset = load_data("data/financial_qa_dataset.csv")

# Tokenize
def preprocess(example):
    model_inputs = tokenizer(example["input_text"], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(example["target_text"], max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned_model",
    evaluation_strategy="no",  # or "steps" if you have eval dataset
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_steps=10,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=5,
    predict_with_generate=True
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./finetuned_model")

