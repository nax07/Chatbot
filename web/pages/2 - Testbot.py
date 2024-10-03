# Import libraries
import streamlit as st
import os
import sys
from langchain import hub
from langchain import HuggingFacePipeline
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig, pipeline
from huggingface_hub import login
from langchain_huggingface.llms import HuggingFacePipeline
import torch

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
modelo_a_link = {
    "gpt2-xl": "openai-community/gpt2-xl",
    "modelo2": "openai-community/gpt2-medium",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct"
}
template = """
You are a question-answering assistant. Answer the question. If you don’t know the answer, simply say you don’t know. Use concise sentences, no more than 3.

Question: {question}

Answer:
"""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


# Inicialize session state
st.session_state.setdefault("idioma", "Inglés")  # Idioma del input y output
st.session_state.setdefault("modelo", "False")   # Nombre corto del modelo
st.session_state.setdefault("process", False)    # llm (HuggingFacePipeline)
st.session_state.setdefault("embeddings", False) # Embeddings
st.session_state.setdefault("messages", [])      # Historial de mensajes
st.session_state.setdefault("retriever", False)  # Retriever
st.session_state.setdefault("key", False)        # Hugging face key with access

####################################### Auxiliar functions #######################################
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def llm_loading(model_id, key=False):
    if key:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=key)
        tokenizer.pad_token = tokenizer.eos_token
    
        config = AutoConfig.from_pretrained(model_id, token=key)
        config.rope_scaling = { "type": "linear", "factor": 8.0 }
    
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=key,
            device_map="auto",
            quantization_config=bnb_config,
        )
    
        text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128
        ) 
    
        hf = HuggingFacePipeline(pipeline=text_generator)
    else:
    
        hf = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            model_kwargs={"temperature": 0.7, "trust_remote_code": True},
            pipeline_kwargs={"max_new_tokens": 100},
        )
    return hf

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
    return output.split("Answer:")[1].strip()

def RAG(question, llm, retriever):
    prompts = hub.pull("rlm/rag-prompt")
    rag_chain = (
        RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
        | prompts
        | llm
        | StrOutputParser()
    )
    output = rag_chain.invoke(question)
    return output.split("Answer:")[1].strip()


####################################### Main App ######################################

st.title("Chatbot_Test")

####################################### Sidebar #######################################

## Settings
st.sidebar.title('Opciones')

# Languages
st.sidebar.subheader('Idioma')
idioma = st.sidebar.selectbox(
    "Selecciona el idioma:",
    ("Inglés", "Español", "Francés",
     "Aleman", "Italiano","Ruso",
     "Chino (Mandarín)", "Árabe", "Hindi"),
)

# LLM
st.sidebar.subheader('Modelo')
model_name = st.sidebar.selectbox(
    "Selecciona el modelo de lenguaje natural:",
    ("gpt2-xl", "modelo2", "Llama-3.1-8B-Instruct"),
)
if model_name == "Llama-3.1-8B-Instruct":
    st.session_state.key = st.sidebar.text_input("Introduce Hugging Face Key", type="password")
    
## Configs
st.sidebar.subheader('Configuraciones del Chatbot')

# Other options

option = st.sidebar.radio(
    "Choose an option:",
    ("Regular processing", "Advanced prompts processing", "Regular RAG", "Multi-Query RAG")
)
#Adv_prompts = st.sidebar.checkbox("Activar Prompts Avanzados")
#RAG = st.sidebar.checkbox("Activar RAG")

# Buttons to confirm configurations
set_button = st.sidebar.button("Confirmar Configuraciones")
clc_historial = st.sidebar.button("Limpiar historial")

## Clean Chat History
if clc_historial:
    st.session_state.messages = []

## Set Configs
if set_button:
    st.session_state.messages = []
    if model_name != st.session_state.modelo:
        st.session_state.modelo = model_name
        modelo = modelo_a_link.get(model_name)
        st.session_state.process = llm_loading(modelo, st.session_state.key)

    
    #if idioma != "Inglés":
    #    lan1 = idioma_a_abreviacion.get(idioma)
    #    st.session_state.lan_en = load_translator(lan1, "en")
    #    st.session_state.en_lan = load_translator("en", lan1)
    
    if option in ["Regular RAG", "Multi-Query RAG"]:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-l6-v2",
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        vectorstore = FAISS.load_local(vectorstore_path, st.session_state.embeddings, allow_dangerous_deserialization=True)
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


####################################### Chatbot #######################################

prompt = st.chat_input(f'Envía un mensaje')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    if st.session_state.process:
        st.session_state.messages.append({"role": "user", "content": prompt})
        if option == "Multi-Query RAG":
            response = RAG(prompt, llm=st.session_state.process, embeddings=st.session_state.embeddings, retriever=st.session_state.retriever)
        elif option == "Regular RAG":
            response = RAG(prompt, llm=st.session_state.process, embeddings=st.session_state.embeddings, retriever=st.session_state.retriever)
        elif option == "Advanced prompts processing":
            response = advanced_processing(prompt, llm=st.session_state.process)
        else:
            response = processing(prompt, llm=st.session_state.process)   
             
            #solution = data_processing(translated_prompt, Adv_prompts, RAG, st.session_state.process, st.session_state.embeddings, st.session_state.vectorstore)
            #response = data_processing(prompt, Adv_prompts, RAG, st.session_state.process, st.session_state.embeddings, st.session_state.vectorstore)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    else:
        st.warning("El modelo no está cargado. Seleccione un modelo y confirme las configuraciones.")
