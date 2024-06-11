import streamlit as st

def Chatbot_app():


    st.title("Chatbot")
    st.write("Opciones:")

    # Initial column for configuration options
    col1, col2 = st.columns(2)

    with col1:
        RAG = st.checkbox("Enable RAG", key="enabled_RAG")
        Adv_prompts = st.checkbox("Enable advanced prompts", key="enabled_prompts")

    with col2:
        option = st.selectbox(
            "Select LLM",
            ("gpt2-medium", "banana phone", "19 $ fornite card"),
            index=None,
            placeholder="Select LLM model..."
        )
    
    # Button to confirm settings
    if st.button("Confirmar Configuraciones"):
        if option == None:
            st.error("Debe seleccionar un modelo LLM.")
        else:
            # Reset the chat history
            st.session_state.messages = []
    # Create space for the chatbot
    st.write(f'Chat{option}')
    prompt = st.chat_input('¿Qué tal?')
    
    

    with st.container(border=True):
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt & option != None:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": prompt})
            st.experimental_rerun()
