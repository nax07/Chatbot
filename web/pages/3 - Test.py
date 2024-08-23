## Import libraries
import streamlit as st
import os
import sys


## Add path for the imports
sys.path.append('/mount/src/chatbot/web/pages/libraries')

from text_processing import *
from text_translation import *


## Inicialize session state
if "idioma" not in st.session_state:
        st.session_state.idioma = "Inglés"

if "modelo" not in st.session_state:
        st.session_state.modelo = "gpt2-medium"

if "lan_en" not in st.session_state:
        st.session_state.lan_en = False
    
if "en_lan" not in st.session_state:
        st.session_state.en_lan = False

if "process" not in st.session_state:
        st.session_state.process = False
    
if "messages" not in st.session_state:
        st.session_state.messages = []


## Variables
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
    "dolly-v2-12b": "databricks/dolly-v2-12b"
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

st.sidebar.subheader('Configuraciones del Chat')
# Opción para activar/desactivar RAG
RAG = st.sidebar.checkbox("Activar RAG", key="enabled_RAG")

# Opción para activar/desactivar prompts avanzados
Adv_prompts = st.sidebar.checkbox("Activar prompts avanzadas", key="enabled_prompts")

# Selección del modelo de lenguaje en la barra lateral
mod_selec = st.sidebar.selectbox(
    "Select LLM",
    ("gpt2-medium", "banana phone", "dolly-v2-7b"),
)

# Botón para confirmar configuraciones

cols = st.columns(2)

with cols[0]:
    set_button = st.sidebar.button("Confirmar Configuraciones")

with cols[1]:
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
        lan2 = "en"
        st.session_state.lan_en = load_translator(lan1, lan2)
        st.session_state.en_lan = load_translator(lan2, lan1)
        
# Create space for the chatbot
prompt = st.chat_input('Envía un mensaje')



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    if st.session_state.process:
        if idioma:   
            if idioma != "Inglés":
                translated_prompt = translator(prompt, st.session_state.lan_en)
                solution = data_processing(translated_prompt, st.session_state.process, RAG, Adv_prompts)
                response = translator(solution, st.session_state.en_lan)
            else:
                response = data_processing(prompt, st.session_state.process, RAG, Adv_prompts)
            
            
            st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "user", "content": modelo})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        else:
            st.warning("Idioma no especificado. Seleccione un idioma y confirme las configuraciones.")
    else:
        st.warning("El modelo no está cargado. Seleccione un modelo y confirme las configuraciones.")
