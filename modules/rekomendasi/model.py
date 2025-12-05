import toml
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

secrets = toml.load(".streamlit/secrets.toml")
GOOGLE_API_KEY = secrets["GOOGLE_API_KEY"]
QDRANT_URL = secrets["QDRANT_URL"]
QDRANT_API_KEY = secrets["QDRANT_API_KEY"]
COLLECTION_NAME = "coffee_review_taste"

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

if __name__ == "__main__":
    pertanyaan = "aku suka kopi yang pahitnya itu nempel di lidah, terus ada rasa coklatnya gitu ada engga?"
    
    print(f"User: {pertanyaan}\n")
    print("Sedang berpikir...")
    
    # Jalankan Chain
    jawaban = rag_chain.invoke(pertanyaan)
    
    print("--- REKOMENDASI ---")
    print(jawaban)