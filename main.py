from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from engine import get_chat_response
from database import ingest_docs, load_existing_db
import shutil
import os
from pydantic import BaseModel
from typing import List, Dict
from langchain_chroma import Chroma
from database import get_local_embeddings

app = FastAPI()
retriever = None

os.makedirs("data", exist_ok=True)

class AskAI(BaseModel):
    query : str
    collection_name : str
    history: List[Dict]

class IngestFile(BaseModel):
    collection_name : str

OS_DATA_DIR = "data"
if not os.path.exists(OS_DATA_DIR):
    os.makedirs(OS_DATA_DIR)

@app.post("/ingest")
async def upload_dan_pelajari(collection_name: str = Form(...), file: UploadFile = File(...)):
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.pptx')):
        raise HTTPException(status_code=400, detail="Hanya mendukung file PDF dan PPTX!")

    file_path = os.path.join(OS_DATA_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return {"error": f"Gagal menyimpan file: {str(e)}"}

    try:
        global retriever
        retriever = ingest_docs(collection_name) 
        return {
            "filename": file.filename,
            "status": "Success",
            "message": "AI sudah selesai membaca dan membuat indeks data."
        }
    except Exception as e:
        return {"error": f"Gagal memproses dokumen: {str(e)}"}

@app.post("/ask")
def asking_ai(data : AskAI):
    try:
        vector_db = Chroma(
            persist_directory='./chroma_db',
            embedding_function=get_local_embeddings(),
            collection_name=data.collection_name
        )

        current_retriever = load_existing_db(data.collection_name)
    except Exception as e:
        return {"error" : "gagal mengakses file atau belum mengupload file"}

    answer = get_chat_response(current_retriever, data.query, data.history)
    return {"answer" : answer}