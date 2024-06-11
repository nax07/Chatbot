from transformers import pipeline,AutoTokenizer, AutoModelForCausalLM

def data_processing(text):

    pipe = pipeline("text-generation", model="openai-community/gpt2-medium")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")

    prompt = f"Question: {text}.\nAnswer:"

    generated_text =  pipe(prompt, max_length=100, num_return_sequences=1, truncation=True)

    generated_text = generated_text[0]["generated_text"].split("\n")[1]#[8:]


    return generated_text
