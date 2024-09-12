# Import libraries
import streamlit as st
import os
import sys
from langchain import hub

# Add path for the imports
sys.path.append('/mount/src/chatbot/web/pages/libraries')

from text_processing import *
from text_translation import *
from RAG import *

# Inicialize session state
st.session_state.setdefault("idioma", "Inglés")
st.session_state.setdefault("modelo", "gpt2-medium")
st.session_state.setdefault("lan_en", False)
st.session_state.setdefault("en_lan", False)
st.session_state.setdefault("process", False)
st.session_state.setdefault("messages", [])

# Variables
idioma_a_abreviacion = {
    "Inglés": "en",
    "Español": "es",
    "Francés": "fr",
    "Alemán": "de",
    "Italiano": "it",
    "Ruso": "ru",
    "Chino (Mandarín)": "zh",
    "Árabe": "ar",
    "Hindi": "hi"
}

modelos = {
    "gpt2": "openai-community/gpt2",
    "Qwen-VL": "Qwen/Qwen-VL-Chat",
    "dolly-v2-7b": "databricks/dolly-v2-12b"
}

## Main App
st.title("Chatbot_Test")

# Sidebar
st.sidebar.title('Opciones')
st.sidebar.subheader('Idioma')
idioma = st.sidebar.selectbox(
    "Seleccionar idioma",
    ("Español", "Inglés", "Francés",
     "Aleman", "Italiano","Ruso",
     "Chino (Mandarín)", "Árabe", "Hindi"),
)

# Selección del modelo de lenguaje en la barra lateral
st.sidebar.subheader('Modelo')
mod_selec = st.sidebar.selectbox(
    "Select LLM",
    ("gpt2-medium", "banana phone", "dolly-v2-7b"),
)

st.sidebar.subheader('Configuraciones del Chat')

# Opción para activar/desactivar prompts avanzados
# Adv_prompts = st.sidebar.checkbox("Activar prompts avanzadas", key="enabled_prompts")

# Opción para activar/desactivar RAG
RAG = st.sidebar.checkbox("Activar RAG")
if RAG:
    chunk_size = st.sidebar.slider("Seleccione el tamaño del chunk:", min_value=10, max_value=1000, value=200)
    chunk_overlap = st.sidebar.slider("Seleccione el solapamiento entre chunks:", min_value=0, max_value=chunk_size, value=30)
    n_docs_retrieved =  st.sidebar.number_input(
        "Ingrese el número de chunks recuperados:", 
        min_value=1, 
        max_value=20, 
        value=5  
    )
    RAG_files = st.sidebar.file_uploader("Sube los archivos de texto para hacer RAG aquí: ", accept_multiple_files=True, type=["txt"])



# Botón para confirmar configuraciones
set_button = st.sidebar.button("Confirmar Configuraciones")
clc_historial = st.sidebar.button("Limpiar historial")

if clc_historial:
    # Reset the chat history
    st.session_state.messages = []

if set_button:
    # Reset the chat history
    st.session_state.messages = []

    # Set selected configurations
    modelo = modelos.get(mod_selec)
    if modelo != st.session_state.process:
        st.session_state.process = model_loading(modelo)
    if idioma != "Inglés":
        lan1 = idioma_a_abreviacion.get(idioma)
        st.session_state.lan_en = load_translator(lan1, "en")
        st.session_state.en_lan = load_translator("en", lan1)
    if RAG and RAG_files:
        all_text = []
        for file in RAG_files:
            string_data = file.read().decode("utf-8") 
            all_text.append(string_data)
        RAG_retriver = RAG_retriever(all_text, chunk_size, chunk_overlap, n_docs_retrieved)
        prompt = hub.pull("rlm/rag-prompt")

# Create space for the chatbot
prompt = st.chat_input(f'Envía un mensaje')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    if st.session_state.process:
            st.session_state.messages.append({"role": "user", "content": prompt})
            if idioma != "Inglés":
                translated_prompt = translator(prompt, st.session_state.lan_en)
                solution = data_processing(translated_prompt, st.session_state.process, RAG, Adv_prompts)
                response = translator(solution, st.session_state.en_lan)
            else:
                response = data_processing(prompt, st.session_state.process, RAG, Adv_prompts)
    
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    else:
        st.warning("El modelo no está cargado. Seleccione un modelo y confirme las configuraciones.")
