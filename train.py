import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq


model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Data Ingestion & Formatting (The Multi-Task Pipeline)
def load_and_merge_data(qa_path, sentiment_path):
    # Load QA Dataset
    df_qa = pd.read_csv(qa_path)
    # Add the command prefix so the AI knows its task
    df_qa["input_text"] = "Answer this finance question: " + df_qa["question"].astype(str)
    df_qa["target_text"] = df_qa["answer"].astype(str)
    df_qa = df_qa[["input_text", "target_text"]].dropna()

    # Load Sentiment Dataset (Assuming columns are 'sentence' and 'sentiment')
    df_sentiment = pd.read_csv(sentiment_path)
    # Add a different command prefix for classification
    df_sentiment["input_text"] = "Classify financial sentiment (Positive/Negative/Neutral): " + df_sentiment["sentence"].astype(str)
    df_sentiment["target_text"] = df_sentiment["sentiment"].astype(str)
    df_sentiment = df_sentiment[["input_text", "target_text"]].dropna()

    # Merge the datasets together
    df_combined = pd.concat([df_qa, df_sentiment], ignore_index=True)
    
    # CRITICAL: Shuffle the data! 
    # If it learns all QA first, then all Sentiment, it will suffer from "catastrophic forgetting".
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return Dataset.from_pandas(df_combined)

# Load your specific files
dataset = load_and_merge_data("data/financial_qa_dataset.csv", "data/sentiment_dataset.csv")

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
    per_device_train_batch_size=4, # Increased slightly if 8GB RAM allows
    num_train_epochs=3,            # 1 epoch isn't enough for multi-tasking
    save_steps=50,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    predict_with_generate=True
)

# 5. Execution
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Stand user activated. Commencing unified training...")
trainer.train()
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
print("Training complete. Model saved.")
