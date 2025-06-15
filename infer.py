from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "NousResearch/TinyLLaMA-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

def generate_answer(question, max_length=150):
    inputs = tokenizer.encode(question, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_p=0.95, temperature=0.8)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    question = input("Enter your finance question: ")
    answer = generate_answer(question)
    print("\nAnswer:\n", answer)
