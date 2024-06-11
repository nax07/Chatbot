import streamlit as st

def Chatbot_app():


    st.title("Chatbot")
    st.write("Puedes preguntar al Chatbot lo que necesites")

    

    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    col1, col2 = st.columns(2)

    with col1:
        RAG = st.checkbox("Enable RAG", key="enabled")
        Adv_prompts = st.checkbox("Enable advanced prompts", key="enabled")
        
    with col2:
        option = st.selectbox(
            "Select LLM",
            ("gpt2-medium", "banana phone", "19 $ fornite card"),
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
        )
    
    with st.container():
        st.write("Chaaat")
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input('¿Qué tal?')
    
