
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainingArguments,
)

from datasets import load_dataset, Dataset
import pandas as pd

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load CSV data into HuggingFace Dataset
def load_data(path):
    df = pd.read_csv(path)
    dataset = Dataset.from_pandas(df)
    return dataset.train_test_split(test_size=0.1)

# Preprocessing function
def preprocess_function(examples):
    inputs = ["Q: " + q for q in examples["question"]]
    targets = [a for a in examples["answer"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def fine_tune_model():
    dataset = load_data("data/financial_qa_dataset.csv")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./finetuned_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=3,
        predict_with_generate=True,
        logging_dir="./logs"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")

if __name__ == "__main__":
    fine_tune_model()


