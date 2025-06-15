import gradio as gr
from utils import load_model, generate_response
import json

# Load config
with open("config.json") as f:
    config = json.load(f)

# Load model and tokenizer
model, tokenizer = load_model(
    config["model_name"],
    quantized=config.get("quantized", True)
)

def chatbot(prompt):
    return generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=config.get("max_length", 512),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.9),
        top_k=config.get("top_k", 50),
        num_beams=config.get("num_beams", 4)
    )

# Gradio UI
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Ask me something about finance..."),
    outputs="text",
    title="Finance Q&A + Sentiment Chatbot",
    description="Ask financial questions or give a sentence to get sentiment + reasoning!"
)

if __name__ == "__main__":
    iface.launch()

