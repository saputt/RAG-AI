import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader

load_dotenv()

def get_local_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    model_kwargs = {"device" : "cpu"}
    encode_kwargs = {"normalize_embeddings" : True}

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def ingest_docs(filepath, collection_name):
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = get_local_embeddings()

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever(search_kwargs={"k" : 10})

def load_existing_db(collection_name):
    embeddings = get_local_embeddings()

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name=collection_name   
    )
    print(f"DEBUG: Ngecek koleksi [{collection_name}], jumlah data: {vectorstore._collection.count()}")
    return vectorstore.as_retriever(search_kwargs={"k" : 8})