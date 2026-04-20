from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

def get_chat_response(retriever, user_input, history):
    llm = ChatOpenAI(
        model="google/gemma-4-26b-a4b-it",
        openai_api_key=os.getenv("OPEN_ROUTER_API"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:8000", 
            "X-Title": "Chatbot RAG Sauki", 
        },
        temperature=0.3,
        max_tokens=2048
    )

    context_chat=""

    for i in history:
        role = "User" if i['role'] == 'USER' else 'ASSISTANT'
        context_chat += f"{role}: {i['content']}\n"

    print(context_chat)

    rephrase_prompt = f"""Berdasarkan percakapan berikut, ubah pertanyaan terbaru user menjadi pertanyaan mandiri (standalone question) yang bisa dipahami tanpa melihat history.
    
    History:
    {context_chat}
    
    Pertanyaan Terbaru: {user_input}
    
    Pertanyaan Mandiri:"""

    standalone_query = llm.invoke(rephrase_prompt).content

    system_prompt = """
        Anda adalah Asisten Dosen yang ahli dan komunikatif. 
        
        TUGAS ANDA:
        Menjelaskan materi secara mendalam dengan menjadikan KONTEKS sebagai referensi UTAMA.

        ATURAN PENGEMBANGAN JAWABAN:
        1. GROUNDING: Jawaban harus dimulai dari fakta yang ada di KONTEKS.
        2. ELABORASI: Anda SANGAT DISARANKAN untuk menambahkan penjelasan, detail, dan contoh tambahan dari pengetahuan Anda agar mahasiswa lebih paham, SELAMA topik tersebut masih relevan dengan isi KONTEKS.
        3. FILTER TOPIK: Jika pertanyaan mahasiswa benar-benar di luar topik materi (misal: tanya resep, gosip, atau hal random), Anda harus menolak dengan sopan dan mengarahkan kembali ke materi.
        4. GAYA MENGAJAR: Gunakan bahasa yang edukatif, santai (seperti asdos ke mahasiswa), dan gunakan analogi jika membantu.
        5. JIKA PERTANYAAN TIDAK ADA DI KONTEKS: Kirim kalimat berikut, "Maaf jawaban tidak tersedia"

        PENTING: 
        Jangan hanya membaca teks. Jadilah asisten yang bisa mengembangkan materi di dokumen menjadi penjelasan yang lebih luas namun tetap di jalurnya.

        KONTEKS:
        {context}

        PERTANYAAN:
        {question}

        JAWABAN:
        """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    rag_chain = (
        {
            "context": lambda x: retriever.invoke(standalone_query),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm 
        | StrOutputParser()
    )

    response = rag_chain.invoke({"question" : user_input})
    return response