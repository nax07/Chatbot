from transformers import pipeline

def translator(text, language1, language2):
    modelo = f"Helsinki-NLP/opus-mt-{language1}-{language2}"
    pipe = pipeline('translation', model=modelo)
    translated_text = pipe(text)[0]['translation_text']
    return translated_text

    
