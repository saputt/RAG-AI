from fastapi import FastAPI, UploadFile, File, HTTPException
from engine import get_chat_response
from database import ingest_docs, load_existing_db
import shutil
import os
from pydantic import BaseModel

app = FastAPI()
retriever = None

os.makedirs("data", exist_ok=True)


class AskAI(BaseModel):
    question : str

OS_DATA_DIR = "data"
if not os.path.exists(OS_DATA_DIR):
    os.makedirs(OS_DATA_DIR)

if  os.path.exists("./chroma_db") and os.listdir("./chroma_db"):
    retriever = load_existing_db()

@app.post("/ingest")
async def upload_dan_pelajari(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Hanya mendukung file PDF!")

    file_path = os.path.join(OS_DATA_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return {"error": f"Gagal menyimpan file: {str(e)}"}

    try:
        global retriever
        retriever = ingest_docs() 
        return {
            "filename": file.filename,
            "status": "Success",
            "message": "AI sudah selesai membaca dan membuat indeks data."
        }
    except Exception as e:
        return {"error": f"Gagal memproses dokumen: {str(e)}"}

@app.get("/ask")
def asking_ai(data : AskAI):
    if not retriever:
        return {"error" : "Uploud file dulu lewat endpoint /ingest"}
    teks = data.question

    answer = get_chat_response(retriever, teks)
    return {"answer" : answer}