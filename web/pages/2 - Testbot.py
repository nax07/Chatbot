# Import libraries
import streamlit as st
import os
import sys
from langchain import hub
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

# Variables
vectorstore_path = "/mount/src/chatbot/web/pages/vectorstore"

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
    "gpt2-xl": "openai-community/gpt2-xl",
    "modelo2": "openai-community/gpt2-medium",
    "modelo3": "openai-community/gpt2-medium"
}

# Inicialize session state
st.session_state.setdefault("idioma", "Inglés")
st.session_state.setdefault("modelo", "gpt2-medium")
st.session_state.setdefault("lan_en", False)
st.session_state.setdefault("en_lan", False)
st.session_state.setdefault("process", False)
st.session_state.setdefault("embeddings", False)
st.session_state.setdefault("vectorstore", False)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pipe", False)

# Auxiliar functions
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def llm_loading(model_name):
    hf = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        model_kwargs={"temperature": 0.7, "trust_remote_code": True},
        pipeline_kwargs={"max_new_tokens": 100},
    )
    return hf

def processing(question, llm):
    return llm.invoke(question)

def advanced_processing(question, llm):
    template = """
    You are a question-answering assistant. Answer the question. If you don’t know the answer, simply say you don’t know. Use concise sentences, no more than 3.

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    adv_chain = (
        RunnableParallel({"question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    output = adv_chain.invoke(question)
    return output.split("Answer:")[1].strip()

def RAG(question, llm, embeddings, vectorstore):
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    output = rag_chain.invoke(question)
    return output.split("Answer:")[1].strip()
    
def data_processing(question, Adv_prompts, RAG, llm, embeddings, vectorstore):
    if Adv_prompts:
        template = """
        You are a question-answering assistant. Answer the question. If you don’t know the answer, simply say you don’t know. Use concise sentences, no more than 3.

        Question: {question}

        Answer:
        """

        prompt = ChatPromptTemplate.from_template(template)
        adv_chain = (
            RunnableParallel({"question": RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
        )
        output = adv_chain.invoke(question)
        return output.split("Answer:")[1].strip()

    elif RAG:
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
        )
        output = rag_chain.invoke(question)
        return output.split("Answer:")[1].strip()
    else:
        return llm.invoke(question)



## Main App
st.title("Chatbot_Test")

# Sidebar
st.sidebar.title('Opciones')
st.sidebar.subheader('Idioma')
idioma = st.sidebar.selectbox(
    "Selecciona el idioma:",
    ("Inglés", "Español", "Francés",
     "Aleman", "Italiano","Ruso",
     "Chino (Mandarín)", "Árabe", "Hindi"),
)

# Selección del modelo de lenguaje en la barra lateral
st.sidebar.subheader('Modelo')
mod_selec = st.sidebar.selectbox(
    "Selecciona el modelo de lenguaje natural:",
    ("gpt2-xl", "modelo2", "modelo3"),
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
    
    # Set selected configurations
    modelo = modelos.get(mod_selec)
    if modelo != st.session_state.process:
        st.session_state.process = llm_loading(modelo)
    if idioma != "Inglés":
        lan1 = idioma_a_abreviacion.get(idioma)
        st.session_state.lan_en = load_translator(lan1, "en")
        st.session_state.en_lan = load_translator("en", lan1)
    if RAG:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-l6-v2",
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        st.session_state.vectorstore = FAISS.load_local(vectorstore_path, st.session_state.embeddings, allow_dangerous_deserialization=True)
    else:
        st.session_state.embeddings = False
        st.session_state.vectorstore = False

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
                solution = data_processing(translated_prompt, Adv_prompts, RAG, st.session_state.process, st.session_state.embeddings, st.session_state.vectorstore)
                response = translator(solution, st.session_state.en_lan)
            else:
                response = data_processing(prompt, Adv_prompts, RAG, st.session_state.process, st.session_state.embeddings, st.session_state.vectorstore)
    
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    else:
        st.warning("El modelo no está cargado. Seleccione un modelo y confirme las configuraciones.")
