from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def model_loading(modelo):
    return pipeline("text-generation", model=modelo, trust_remote_code=True, return_full_text=True)

def data_processing(text, pipe, RAG=False, Adv_prompts=False, max_len=100):
    prompt = f"Question: {text}.\nAnswer:"
    generated_text = pipe(prompt, num_return_sequences=1, truncation=True)
    return generated_text[0]["generated_text"].split("\n")[1].strip()
