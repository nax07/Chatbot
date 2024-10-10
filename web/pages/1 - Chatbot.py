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
from langchain_together import Together
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

####################################### Variables #######################################
vectorstore_path = "/mount/src/chatbot/web/pages/vectorstore"
modelo_a_link = {
    "Gpt2-xl": "openai-community/gpt2-xl",
    "Cohere": "cohere",
    "Llama3": "meta-llama/Llama-3.2-3B-Instruct-Turbo"
}
template = """
You are a question-answering assistant. Answer the question. If you don’t know the answer, simply say you don’t know. Use concise sentences, no more than 3.

Question: {question}

Answer:
"""
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

# Inicialize session state
st.session_state.setdefault("model_name", False)       # Model name
st.session_state.setdefault("model_llm", False)        # Model LLM
st.session_state.setdefault("model_key", False)        # Model API key with access
st.session_state.setdefault("idioma", "Inglés")        # Idioma
st.session_state.setdefault("modelo_en_lan", False)
st.session_state.setdefault("modelo_lan_en", False)

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

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

def llm_loading(model_id, key=False):
    if model_id == "cohere":
        return Cohere(cohere_api_key=key, max_tokens=265)
    elif model_id == "meta-llama/Llama-3.2-3B-Instruct-Turbo":
        return Together(model="meta-llama/Llama-3.2-3B-Instruct-Turbo", together_api_key=key)
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
    rag_chain = (
        RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
        | prompts
        | llm
        | StrOutputParser()
    )
    output = rag_chain.invoke(question)
    return output, retrieved_docs #output.split("Answer:")[1].strip()

def Multi_Query(question, llm, retriever):

    template = """You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoque({"question": question})

    prompt = hub.pull("rlm/rag-prompt")
    final_chain = (
        {"context": retrieval_chain,
        "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return final_chain.invoke({"question": question})

def load_translator(language1, language2):
    modelo = f"Helsinki-NLP/opus-mt-{language1}-{language2}"
    return pipeline('translation', model=modelo)

def translator(text, pipe):
    return pipe(text)[0]['translation_text']

####################################### Main App ######################################

st.title("Chatbot")

####################################### Sidebar #######################################

## Settings
st.sidebar.title('Opciones')

# Languages
st.session_state.idioma = st.sidebar.selectbox(
    "Selecciona el idioma:",
    ("Español", "Inglés", "Francés",
     "Aleman", "Italiano","Ruso",
     "Chino (Mandarín)", "Árabe", "Hindi"),
)

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
    "Tipo de procesamiento:",
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
    if st.session_state.idioma != "Inglés":
        st.session_state.modelo_en_lan = load_translator("en", idioma_a_abreviacion.get(st.session_state.idioma))
        st.session_state.modelo_lan_en = load_translator(idioma_a_abreviacion.get(st.session_state.idioma), "en")
    
####################################### Chatbot #######################################

prompt = st.chat_input(f'Envía un mensaje')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    if st.session_state.process:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if st.session_state.idioma != "Inglés":
            input = translator(prompt, st.session_state.modelo_lan_en)
        else:
            input = prompt    
        
        if option == "Multi-Query RAG":
            response = Multi_Query(input, llm=st.session_state.process, retriever=st.session_state.retriever)
            #response, docs = RAG_test(input, llm=st.session_state.process, retriever=st.session_state.retriever)
            #st.session_state.messages.append({"role": "user", "content": docs})
        elif option == "Regular RAG":
            response = RAG(input, llm=st.session_state.process, retriever=st.session_state.retriever)
        elif option == "Advanced prompts processing":
            response = advanced_processing(input, llm=st.session_state.process)
        else:
            response = processing(input, llm=st.session_state.process)   

        if st.session_state.idioma != "Inglés":
            response =  translator(response, st.session_state.modelo_en_lan)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    else:
        st.warning("El modelo no está cargado. Seleccione un modelo y confirme las configuraciones.")
