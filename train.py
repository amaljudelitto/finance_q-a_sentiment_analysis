from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os

model_name = "NousResearch/TinyLLaMA-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def fine_tune_model():
    dataset = load_dataset("data/financial_qa_dataset.txt", tokenizer)

    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=1,
        logging_dir='./logs',
        logging_steps=50
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")

if __name__ == "__main__":
    fine_tune_model()


