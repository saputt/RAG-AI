from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

def get_chat_response(retriever, user_input):
    llm = ChatOpenAI(
        model="qwen/qwen3.6-plus:free",
        openai_api_key=os.getenv("OPEN_ROUTER_API"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:8000", 
            "X-Title": "Chatbot RAG Sauki", 
        },
        temperature=0.7,
        max_tokens=2048
    )

    system_prompt = """
        Anda adalah asisten dosen yang ahli dan komunikatif. 
        Tugas Anda adalah menjelaskan materi berdasarkan konteks dokumen yang diberikan.

        ATURAN:
        1. Jika di dalam KONTEKS hanya ada poin-poin singkat, tugas Anda adalah MENJELASKANNYA secara mendalam dan mudah dimengerti.
        2. Gunakan pengetahuan internal Anda untuk melengkapi penjelasan tersebut, asalkan topiknya masih relevan dengan KONTEKS.
        3. Jika pertanyaan sama sekali tidak ada hubungannya dengan KONTEKS, baru Anda katakan tidak tahu.
        4. Jawab dalam Bahasa Indonesia yang santai tapi edukatif.
        5. Jika jawaban atau pertanyaan tidak terdapat di pengetahun internal respon seperti ini saja "Maaf hal tersebut tidak tersedia pada pengetahuan internal"

        KONTEKS MATERI:
        {context}

        PERTANYAAN MAHASISWA:
        {question}

        JAWABAN DETAIL:
        """



    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm 
        | StrOutputParser()
    )

    response = rag_chain.invoke(user_input)
    return response