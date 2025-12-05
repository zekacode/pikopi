from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- KONFIGURASI ---
GOOGLE_API_KEY = "AIzaSyAdP2wPK0n-G03yO1YE60yE2NPMBo9sJSQ"
QDRANT_URL = "https://cd6d1bd4-f86a-4486-9b88-5d5870e152a5.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.erRi2rr3j-9QYSvjsMiQoc05EL4e-lLH5Qtu4Xwr92c"
COLLECTION_NAME = "coffee_review_taste"

# 1. Setup Components
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


retriever = vector_store.as_retriever(search_kwargs={'k': 3})


template = """
Kamu adalah asisten Barista yang ramah.
Gunakan data kopi berikut untuk menjawab permintaan user.

DATA KOPI:
{context}

PERTANYAAN USER:
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

# --- TEST ---
if __name__ == "__main__":
    pertanyaan = "aku suka kopi yang pahitnya itu nempel di lidah, terus ada rasa coklatnya gitu ada engga?"
    
    print(f"User: {pertanyaan}\n")
    print("Sedang berpikir...")
    
    # Jalankan Chain
    jawaban = rag_chain.invoke(pertanyaan)
    
    print("--- REKOMENDASI ---")
    print(jawaban)