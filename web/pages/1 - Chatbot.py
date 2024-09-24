# Import libraries
import streamlit as st
import os
import sys
from langchain import hub

# Add path for the imports
sys.path.append('/mount/src/chatbot/web/pages/libraries')

from text_processing import *
from text_translation import *

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
    "gpt2-medium": "openai-community/gpt2-medium",
    "modelo2": "openai-community/gpt2-medium",
    "modelo3": "openai-community/gpt2-medium"
}

vectorstore_path = "vectorstore"

## Main App
st.title("Chatbot_Test")

# Sidebar
st.sidebar.title('Opciones')
st.sidebar.subheader('Idioma')
idioma = st.sidebar.selectbox(
    "Selecciona el idioma:",
    ("Español", "Inglés", "Francés",
     "Aleman", "Italiano","Ruso",
     "Chino (Mandarín)", "Árabe", "Hindi"),
)

# Selección del modelo de lenguaje en la barra lateral
st.sidebar.subheader('Modelo')
mod_selec = st.sidebar.selectbox(
    "Selecciona el modelo de lenguaje natural:",
    ("gpt2-medium", "modelo2", "modelo3"),
)

st.sidebar.subheader('Configuraciones del Chatbot')

# Prompts avanzados
Adv_prompts = st.sidebar.checkbox("Activar Prompts Avanzados")

# Opción para activar/desactivar RAG
RAG = st.sidebar.checkbox("Activar RAG")

# Botón para confirmar configuraciones
set_button = st.sidebar.button("Confirmar Configuraciones")
clc_historial = st.sidebar.button("Limpiar historial")

if clc_historial:
    # Reset the chat history
    st.session_state.messages = []

if set_button:
    # Reset the chat history
    st.session_state.messages = []

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    # Set selected configurations
    modelo = modelos.get(mod_selec)
    if modelo != st.session_state.process:
        st.session_state.process = llm_loading(modelo)
    if idioma != "Inglés":
        lan1 = idioma_a_abreviacion.get(idioma)
        st.session_state.lan_en = load_translator(lan1, "en")
        st.session_state.en_lan = load_translator("en", lan1)

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
                solution = data_processing(translated_prompt, Adv_prompts, RAG, st.session_state.process, embeddings, vectorstore)
                response = translator(solution, st.session_state.en_lan)
            else:
                response = data_processing(prompt, st.session_state.process, RAG, Adv_prompts)
    
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    else:
        st.warning("El modelo no está cargado. Seleccione un modelo y confirme las configuraciones.")
