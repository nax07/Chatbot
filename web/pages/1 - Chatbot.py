import streamlit as st
import os
import sys

sys.path.append('/mount/src/chatbot/web/pages/libraries')

from text_processing import *
from text_translation import *

idioma_a_abreviacion = {
    "Español": "es",
    "Inglés": "en",
    "Francés": "fr",
    "Alemán": "de",
    "Italiano": "it",
    "Ruso": "ru",
    "Chino (Mandarín)": "zh",
    "Árabe": "ar",
    "Hindi": "hi"
}

st.title("Chatbot")

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
modelo = st.sidebar.selectbox(
    "Select LLM",
    ("gpt2-medium", "banana phone", "19 $ fornite card"),
)

# Botón para confirmar configuraciones
set_button = st.sidebar.button("Confirmar Configuraciones / Limpiar historial")
if set_button:
    # Reset the chat history
    st.session_state.messages = []
        
# Create space for the chatbot
prompt = st.chat_input('Envía un mensaje')

if "messages" not in st.session_state:
        st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    
    if idioma != "Inglés":
        lan1 = idioma_a_abreviacion.get(idioma)
        lan2 = "en"
        st.write(f"{lan1}")
        translated_prompt = translator(prompt, lan1, lan2)
        retranslated_prompt = translator(prompt, lan2, lan1)
    else:
        translated_prompt = prompt
        retranslated_prompt = propmt
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": translated_prompt})
    st.session_state.messages.append({"role": "assistant", "content": retranslated_prompt})
    st.rerun()
