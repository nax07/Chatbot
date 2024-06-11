import streamlit as st
from libraries.text_translation import *
from libraries.text_processing import *

def Chatbot_app():


    st.title("Chatbot")
    st.write("Puedes preguntar al Chatbot lo que necesites")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input('¿Qué tal?')
    
