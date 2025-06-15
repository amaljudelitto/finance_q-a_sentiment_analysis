import json
from utils import load_model, generate_response

# Load config
with open("config.json") as f:
    config = json.load(f)

# Load model and tokenizer
model, tokenizer = load_model(
    config["model_name"],
    quantized=config.get("quantized", True)
)

def infer(prompt):
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=config.get("max_length", 512),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.9),
        top_k=config.get("top_k", 50),
        num_beams=config.get("num_beams", 4)
    )
    print("\n[Model Output]:\n", response)


if __name__ == "__main__":
    prompt = input("Enter a financial question or sentence: ")
    infer(prompt)

