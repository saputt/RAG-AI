from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from engine import get_chat_response
from database import ingest_docs, load_existing_db
import shutil
import os
from pydantic import BaseModel
from typing import List, Dict
from langchain_chroma import Chroma
from database import get_local_embeddings
from dotenv import load_dotenv

app = FastAPI()
retriever = None

load_dotenv()

os.makedirs("data", exist_ok=True)

class AskAI(BaseModel):
    query : str
    collection_name : str
    history: List[Dict]
    room_id: str

class IngestFile(BaseModel):
    collection_name : str

class DeleteRoom(BaseModel):
    collection_name : str
    file_path : List[str]

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
        current_retriever = ingest_docs(file_path, collection_name) 

        if not current_retriever:
             return {"error" : "Koleksi tidak ditemukan atau kosong"}

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
        current_retriever = load_existing_db(data.collection_name)
    except Exception as e:
        return {"error" : "gagal mengakses file atau belum mengupload file"}

    answer = get_chat_response(current_retriever, data.query, data.history)
    return {"answer" : answer}

@app.delete("/room")
def delete_room_file(data : DeleteRoom):
    try:
        vector_db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=get_local_embeddings(),
            collection_name=data.collection_name
        )

        vector_db.delete_collection()

        deleted_file = []

        for path in data.file_path:
            if os.path.exists(path):
                deleted_file.append(path)
                os.remove(path)

        return {
            "status" : "Success",
            "chroma_deleted" : data.collection_name,
            "physical_deleted" : deleted_file,
            "message" : "Delete room with fiel successfuly"
        }
    except Exception as e:
        return {"status" : "error", "message" : str(e)}