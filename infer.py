from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# CRITICAL: Point to your locally trained model, not the base HuggingFace model!
model_path = "./finetuned_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.eval()

def generate_answer(prompt, task_type="qa", max_length=150):
    # Dynamic Prefix Routing
    if task_type == "qa":
        formatted_prompt = f"Answer this finance question: {prompt}"
    elif task_type == "sentiment":
        formatted_prompt = f"Classify financial sentiment (Positive/Negative/Neutral): {prompt}"
    else:
        formatted_prompt = prompt

    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_p=0.95, temperature=0.8)
        
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# This allows you to test the model directly in the terminal without running the API
if __name__ == "__main__":
    print("--- Finance Multi-Task AI Activated ---")
    mode = input("Select mode (1 for Q&A, 2 for Sentiment): ")
    user_input = input("Enter your text: ")
    
    task = "qa" if mode == "1" else "sentiment"
    answer = generate_answer(user_input, task_type=task)
    print(f"\n[Output]: {answer}")
