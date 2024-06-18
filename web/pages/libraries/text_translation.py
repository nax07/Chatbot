from transformers import pipeline

def load_translator(language1, language2):
    modelo = f"Helsinki-NLP/opus-mt-{language1}-{language2}"
    return pipeline('translation', model=modelo)

def translator(text, pipe):
    return pipe(text)[0]['translation_text']

    
