from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
def RAG_retriever(list_of_strings, chunk_size=200, chunk_overlap=30, n_docs_retrieved=5):
    # Loads The documents
    documents = [Document(page_content=text) for text in list_of_strings]

    # Splits them into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(documents)

    # Create the embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    # Create the vectorstore & retriever
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": n_docs_retrieved})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
