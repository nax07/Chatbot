import streamlit as st
import os

os.chdir('/mount/src/chatbot')
os.chdir('web/pages/libraries')
cwd = os.getcwd()
st.write(f"{os.listdir(cwd)}")

from text_processing import *
from text_translation import *

st.title("Chatbot")

# Sidebar
st.sidebar.title('Opciones')

st.sidebar.subheader('Idioma')
option = st.sidebar.selectbox(
    "Seleccionar idioma",
    ("Español", "Inglés", "Francés",
     "Portugués", "Aleman", "Italiano",
     "Ruso", "Chino (Mandarín)", "Árabe", "Hindi"),
)

st.sidebar.subheader('Configuraciones del Chat')
# Opción para activar/desactivar RAG
RAG = st.sidebar.checkbox("Activar RAG", key="enabled_RAG")

# Opción para activar/desactivar prompts avanzados
Adv_prompts = st.sidebar.checkbox("Activar prompts avanzadas", key="enabled_prompts")

# Selección del modelo de lenguaje en la barra lateral
option = st.sidebar.selectbox(
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

    st.write(option)
    if option != "Inglés":
        translated_prompt = translator(text, option, "Inglés")
    else:
        translated_prompt = prompt
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": translated_prompt})
    st.experimental_rerun()
