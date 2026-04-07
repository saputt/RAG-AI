import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def ingest_docs():
    loader = DirectoryLoader('./data', glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", task_type="retrieval_query")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever(search_kwargs={"k" : 5})

def load_existing_db():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", task_type="retrieval_query")
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings   
    )
    return vectorstore.as_retriever(search_kwargs={"k" : 5})