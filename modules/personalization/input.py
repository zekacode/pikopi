import time
import toml
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

secrets = toml.load(".streamlit/secrets.toml")
GOOGLE_API_KEY = secrets["GOOGLE_API_KEY"]
QDRANT_URL = secrets["QDRANT_URL"]
QDRANT_API_KEY = secrets["QDRANT_API_KEY"]
COLLECTION_NAME = "coffee_review_taste"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    task_type="retrieval_document"
)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

print("Membaca CSV...")
loader = CSVLoader(
    file_path="Database\coffee_analysis.csv", 
    encoding="utf-8",
    source_column="name" 
)
documents = loader.load()
print(f"Total dokumen dimuat: {len(documents)}")


BATCH_SIZE = 10
SLEEP_TIME = 3 

print("Mulai upload ke Qdrant via LangChain...")

for i in range(0, len(documents), BATCH_SIZE):
    batch_docs = documents[i : i + BATCH_SIZE]
    
    try:
        vector_store.add_documents(batch_docs)
        
        print(f"--> Batch {i} - {i+len(batch_docs)} sukses!")
        
        # Istirahat
        if (i + BATCH_SIZE) < len(documents):
            time.sleep(SLEEP_TIME)
            
    except Exception as e:
        print(f"--> GAGAL di batch {i}: {e}")
        time.sleep(10) 

print("\nSELESAI! Semua data masuk.")