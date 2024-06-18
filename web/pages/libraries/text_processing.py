from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def data_processing(text, modelo, RAG=False, Adv_prompts=False):
    pipe = pipeline("text-generation", model=modelo)
    
    # Construct prompt based on RAG and Adv_prompts flags
    prompt = f"Question: {text}.\nAnswer:"
    
    # Generate text using the pipeline
    generated_text = pipe(prompt, max_length=100, num_return_sequences=1, truncation=True)
    
    # Extract the generated answer from the result
    #generated_text = generated_text[0]["generated_text"].split("\n")[1].strip()
    
    return generated_text
