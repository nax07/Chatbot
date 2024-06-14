from transformers import pipeline

def identify_language(text):
    """ Loads language detection model.
        Source:https://huggingface.co/papluca/xlm-roberta-base-language-detection
        Then, identifies text language using the pipeline given

    Returns:
        _type_: results of the text classification pipeline
    """
    model = "papluca/xlm-roberta-base-language-detection"
    pipe = pipeline("text-classification", model=model)
    return pipe(text, top_k=1, truncation=True)[0]["label"]


def translators(language):
    """Loads translator from language selected to english
       and from selected language to english
       based on the Helsinki-NLP/opus-mt NLP models

    Args:
        language (String): language label

    Returns:
        _type_: returns pipelines for text translation
                from language-english and english-language
    """

    translation_lang_en = f"Helsinki-NLP/opus-mt-{language}-en"
    translation_en_lang = f"Helsinki-NLP/opus-mt-en-{language}"
    pipe_lang_en = pipeline('translation', model=translation_lang_en)
    pipe_en_lang = pipeline('translation', model=translation_en_lang)
    return [pipe_lang_en, pipe_en_lang]

def translator_exec(text, translator):
    """Translates text using translator to english

    Returns:
        String: Translated text
    """

    translated_text = translator(text, max_length=500)[0]['translation_text']
    return translated_text

idioma_a_abreviacion = {
    "Español": "es",
    "Inglés": "en",
    "Francés": "fr",
    "Portugués": "pt",
    "Alemán": "de",
    "Italiano": "it",
    "Ruso": "ru",
    "Chino (Mandarín)": "zh",
    "Árabe": "ar",
    "Hindi": "hi"
}

def translator(text, language1, language2):

    lan1 = idioma_a_abreviacion.get(language1)
    lan2 = idioma_a_abreviacion.get(language2)

    modelo = f"Helsinki-NLP/opus-mt-{lan1}-{lan2}"

    return modelo
    #pipe = pipeline('translation', model=modelo)
    #translated_text = translator(text, max_length=5*len(text))[0]['translation_text']
    #return translated_text

    
