import os
import toml
import streamlit as st
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- SETUP PATH & SECRETS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR)) # Naik ke root 'pikopi'
SECRETS_PATH = os.path.join(ROOT_DIR, ".streamlit", "secrets.toml")

GOOGLE_API_KEY = None
QDRANT_URL = None
QDRANT_API_KEY = None

try:
    if os.path.exists(SECRETS_PATH):
        secrets = toml.load(SECRETS_PATH)
        GOOGLE_API_KEY = secrets["GOOGLE_API_KEY"]
        QDRANT_URL = secrets["QDRANT_URL"]
        QDRANT_API_KEY = secrets["QDRANT_API_KEY"]
    else:
        # Fallback ke Streamlit Cloud Secrets
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        QDRANT_URL = st.secrets["QDRANT_URL"]
        QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
except Exception as e:
    print(f"Error loading secrets: {e}")

COLLECTION_NAME = "coffee_review_taste"

def get_rag_chain():
    """
    Fungsi untuk membuat RAG Chain.
    Dipanggil oleh UI (pages/personalization.py).
    """
    if not GOOGLE_API_KEY or not QDRANT_URL:
        raise ValueError("API Keys belum terload dengan benar.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query" 
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7
    )

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    retriever = vector_store.as_retriever(search_kwargs={'k': 5}) 

    template = """
    Kamu adalah asisten Barista yang ramah dan sangat ahli tentang kopi.
    Tugasmu adalah merekomendasikan biji kopi berdasarkan DATABASE yang diberikan.
    
    INSTRUKSI KHUSUS:
    1. Analisa permintaan user yang berisi detail rasa dan intensitasnya.
    2. Cocokkan dengan DATA KOPI di bawah.
    3. Jelaskan kenapa kamu memilih kopi tersebut (hubungkan intensitas rasa yang diminta user dengan deskripsi kopi).
    4. Jika user meminta "Fruity", cari kopi dengan deskripsi buah-buahan. Jika "Acidity", cari yang deskripsinya asam segar/bright.
    
    DATA KOPI:
    {context}

    PERMINTAAN SPESIFIK USER:
    {question}

    JAWABAN (Saran Kopi Terbaik):
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain