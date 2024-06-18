from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def model_loading(modelo):
    return pipeline("text-generation", model=modelo)

def data_processing(text, pipe, RAG=False, Adv_prompts=False):
    prompt = f"Question: {text}.\nAnswer:"
    generated_text = pipe(prompt, max_length=100, num_return_sequences=1, truncation=True)
    return generated_text[0]["generated_text"].split("\n")[1].strip()
