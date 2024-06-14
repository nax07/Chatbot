from transformers import pipeline,AutoTokenizer, AutoModelForCausalLM

def data_processing(text, modelo, RAG, Adv_prompts):

    pipe = pipeline("text-generation", model=modelo)
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    model = AutoModelForCausalLM.from_pretrained(modelo)
    prompt = f"Question: {text}.\nAnswer:"
    
    if RAG:
        prompt = f"Question: {text}.\nAnswer:"

    if Adv_prompts:
        prompt = f"Question: {text}.\nAnswer:"    
    
    generated_text =  pipe(prompt, max_length=100, num_return_sequences=1, truncation=True)
    generated_text = generated_text[0]["generated_text"].split("\n")[1].strip()


    return generated_text
