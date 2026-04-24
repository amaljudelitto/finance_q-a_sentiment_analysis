import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

# 1. Initialize the Foundation
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Autonomous Cloud Data Ingestion
def load_professional_data():
    print("Downloading enterprise datasets from Hugging Face Hub...")
    
    # Dataset A: Finance Alpaca (Q&A) - We take 1,000 rows so it trains quickly
    ds_qa = load_dataset("gbharti/finance-alpaca", split="train[:1000]")
    df_qa = ds_qa.to_pandas()
    # Combine the 'instruction' and 'input' columns to form the full question
    df_qa["input_text"] = "Answer this finance question: " + df_qa["instruction"].astype(str) + " " + df_qa["input"].astype(str)
    df_qa["target_text"] = df_qa["output"].astype(str)
    df_qa = df_qa[["input_text", "target_text"]]

    # Dataset B: Twitter Financial News (Sentiment) - We take 1,000 rows
    ds_sent = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train[:1000]")
    df_sent = ds_sent.to_pandas()
    # The dataset uses numbers for labels: 0 (Bearish), 1 (Bullish), 2 (Neutral)
    label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
    df_sent["input_text"] = "Classify financial sentiment (Positive/Negative/Neutral): " + df_sent["text"].astype(str)
    df_sent["target_text"] = df_sent["label"].map(label_map).astype(str)
    df_sent = df_sent[["input_text", "target_text"]]

    # Merge and Shuffle to prevent Task Bleed
    df_combined = pd.concat([df_qa, df_sent], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return Dataset.from_pandas(df_combined)

dataset = load_professional_data()

# 3. Tokenization
def preprocess(example):
    model_inputs = tokenizer(example["input_text"], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(example["target_text"], max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# 4. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned_model",
    eval_strategy="no",
    per_device_train_batch_size=4, 
    num_train_epochs=3,            
    save_steps=50,
    save_total_limit=1,
    logging_steps=10,
    predict_with_generate=True
)

# 5. Execution
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer, # Using the modernized 2026 HuggingFace syntax
    data_collator=data_collator,
)

print("Data loaded successfully. Commencing deep learning...")
trainer.train()
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
print("Training complete. Enterprise model saved.")
