import streamlit as st

def Chatbot_app():


    st.title("Chatbot")
    st.write("Opciones:")

    # Initial column for configuration options
    col1, col2 = st.columns(2)

    with col1:
        RAG = st.checkbox("Enable RAG", key="enabled_RAG")
        Adv_prompts = st.checkbox("Enable advanced prompts", key="enabled_prompts")
        option = st.selectbox(
            "Select LLM",
            ("gpt2-medium", "banana phone", "19 $ fornite card")
        )

        # Button to confirm settings
        if st.button("Confirmar Configuraciones"):
            # Reset the chat history
            st.session_state.messages = []

            # Display selected options
            with col2:
                st.write(f"RAG: {'Enabled' if RAG else 'Disabled'}")
                st.write(f"Advanced Prompts: {'Enabled' if Adv_prompts else 'Disabled'}")
                st.write(f"Selected LLM: {option}")

            # Create space for the chatbot
            st.write('Chat')
            if "messages" not in st.session_state:
                st.session_state.messages = []
        
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            prompt = st.chat_input('¿Qué tal?')
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.experimental_rerun()
