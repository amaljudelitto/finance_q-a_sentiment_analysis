import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_model(model_name, quantized=True):
    print(f"Loading model: {model_name} | Quantized: {quantized}")
    if quantized:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9, top_k=50, num_beams=4):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=True,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

