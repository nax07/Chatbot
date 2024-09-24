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

def data_processing(question, Adv_prompts, RAG, llm, embeddings):
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
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        prompt = hub.pull("rlm/rag-prompt")
        llm = llm_loading(model_name)
        rag_chain = (
            RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain.invoke(question)
    else:
        return llm.invoke(question)
