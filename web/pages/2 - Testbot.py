# Import libraries
import streamlit as st
import os
import sys
from langchain import hub, HuggingFacePipeline
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import Cohere
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
from huggingface_hub import login
from langchain_huggingface.llms import HuggingFacePipeline
import torch

####################################### Variables #######################################
vectorstore_path = "/mount/src/chatbot/web/pages/vectorstore"
modelo_a_link = {
    "Gpt2-xl": "openai-community/gpt2-xl",
    "Cohere": "cohere",
    "Llama3": "meta-llama/Llama-3.1-8B-Instruct"
}
template = """
You are a question-answering assistant. Answer the question. If you don’t know the answer, simply say you don’t know. Use concise sentences, no more than 3.

Question: {question}

Answer:
"""

# Inicialize session state
st.session_state.setdefault("model_name", False)       # Model name
st.session_state.setdefault("model_llm", False)        # Model LLM
st.session_state.setdefault("model_key", False)        # Model API key with access

st.session_state.setdefault("embeddings", False)       # Embeddings
st.session_state.setdefault("retriever", False)        # Retriever

st.session_state.setdefault("messages", [])            # Messages history

st.session_state.setdefault("modelo", False)           # Nombre corto del modelo
st.session_state.setdefault("process", False)          # llm (HuggingFacePipeline)

# load embeddings and retriever
if not st.session_state.embeddings or not st.session_state.retriever:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    vectorstore = FAISS.load_local(vectorstore_path, st.session_state.embeddings, allow_dangerous_deserialization=True)
    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

####################################### Auxiliar functions #######################################
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def llm_loading(model_id, key=False):
    if model_id == "cohere":
        return Cohere(cohere_api_key=key, max_tokens=265)
    elif model_id == "":
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=key)
        tokenizer.pad_token = tokenizer.eos_token
    
        config = AutoConfig.from_pretrained(model_id, token=key)
        config.rope_scaling = { "type": "linear", "factor": 8.0 }
    
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=key,
            device_map="auto",
            #quantization_config=bnb_config,
        )
    
        text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128
        ) 
    
        return HuggingFacePipeline(pipeline=text_generator)
    else:
        return HuggingFacePipeline.from_model_id(
                    model_id=model_id,
                    task="text-generation",
                    model_kwargs={"temperature": 0.7, "trust_remote_code": True},
                    pipeline_kwargs={"max_new_tokens": 100})


def processing(question, llm):
    return llm.invoke(question)

def advanced_processing(question, llm):
    prompts = ChatPromptTemplate.from_template(template)
    adv_chain = (
        RunnableParallel({"question": RunnablePassthrough()})
        | prompts
        | llm
        | StrOutputParser()
    )
    output = adv_chain.invoke(question)
    return output  #output.split("Answer:")[1].strip()

def RAG(question, llm, retriever):
    prompts = hub.pull("rlm/rag-prompt")
    rag_chain = (
        RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
        | prompts
        | llm
        | StrOutputParser()
    )
    output = rag_chain.invoke(question)
    return output #output.split("Answer:")[1].strip()

def RAG_test(question, llm, retriever):
    prompts = hub.pull("rlm/rag-prompt")
    retrieved_docs = retriever.invoke(question)
    formatted_docs = format_docs(retrieved_docs)
    rag_chain = (
        RunnableParallel({"context": RunnablePassthrough(), "question": RunnablePassthrough()})
        | prompts
        | llm
        | StrOutputParser()
    )
    output = rag_chain.invoke(formatted_docs, question)
    return output, formatted_docs

####################################### Main App ######################################

st.title("Chatbot_Test")

####################################### Sidebar #######################################

## Settings
st.sidebar.title('Opciones')

# Languages

# LLM
st.sidebar.subheader('Modelo')
st.session_state.model_name = st.sidebar.selectbox(
    "Selecciona el modelo de lenguaje natural:",
    ("Gpt2-xl", "Cohere", "Llama3"),
)
if st.session_state.model_name in ["Llama3", "Cohere"]:
    st.session_state.key = st.sidebar.text_input("Introduce llm Key", type="password")
    
## Configs
st.sidebar.subheader('Configuraciones del Chatbot')

# Other options

option = st.sidebar.radio(
    "Choose an option:",
    ("Regular processing", "Advanced prompts processing", "Regular RAG", "Multi-Query RAG")
)

# Buttons to confirm configurations
set_button = st.sidebar.button("Confirmar Configuraciones")
clc_historial = st.sidebar.button("Limpiar historial")

## Clean Chat History
if clc_historial:
    st.session_state.messages = []

## Set Configs
if set_button:
    st.session_state.messages = []
        
    if st.session_state.model_name in ["Cohere", "Llama3"] and not st.session_state.key:
        st.warning("Falta poner la huggingface key.")

    else:
        modelo = modelo_a_link.get(st.session_state.model_name)
        st.session_state.process = llm_loading(modelo, st.session_state.key)

####################################### Chatbot #######################################

prompt = st.chat_input(f'Envía un mensaje')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    if st.session_state.process:
        st.session_state.messages.append({"role": "user", "content": prompt})
        if option == "Multi-Query RAG":
            response, docs = RAG_test(prompt, llm=st.session_state.process, retriever=st.session_state.retriever)
            st.session_state.messages.append({"role": "user", "content": string(docs)})
        elif option == "Regular RAG":
            response = RAG(prompt, llm=st.session_state.process, retriever=st.session_state.retriever)
        elif option == "Advanced prompts processing":
            response = advanced_processing(prompt, llm=st.session_state.process)
        else:
            response = processing(prompt, llm=st.session_state.process)   
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    else:
        st.warning("El modelo no está cargado. Seleccione un modelo y confirme las configuraciones.")
