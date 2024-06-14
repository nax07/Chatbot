import streamlit as st

st.title("Chatbot")

# Sidebar
st.sidebar.title('Configuraciones')

option = st.sidebar.selectbox(
    "Select Languaje",
    ("Español", "Inglés", "Francés",
    "Aleman", "Chino (Mandarín)", "Árabe"),
)

# Opción para activar/desactivar RAG
RAG = st.sidebar.checkbox("Enable RAG", key="enabled_RAG")

# Opción para activar/desactivar prompts avanzados
Adv_prompts = st.sidebar.checkbox("Enable advanced prompts", key="enabled_prompts")

# Selección del modelo de lenguaje en la barra lateral
option = st.sidebar.selectbox(
    "Select LLM",
    ("gpt2-medium", "banana phone", "19 $ fornite card"),
)

# Botón para confirmar configuraciones
set_button = st.sidebar.button("Confirmar Configuraciones")
if set_button:
    # Reset the chat history
    st.session_state.messages = []
        
# Create space for the chatbot
prompt = st.chat_input('¿Qué tal?')


with st.container(border=True):

    if "messages" not in st.session_state:
            st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": prompt})
        st.experimental_rerun()
