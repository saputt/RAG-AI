
import sys
print("Starting imports...")
import fastapi
print("Fastapi imported")
import langchain
print("Langchain imported")
from langchain_chroma import Chroma
print("Chroma imported")
from langchain_huggingface import HuggingFaceEmbeddings
print("HuggingFaceEmbeddings imported")
from database import get_local_embeddings
print("Database local embeddings imported")
print("All imports successful")
