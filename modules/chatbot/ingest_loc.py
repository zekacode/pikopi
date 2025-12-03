import os
import time
import toml
import pandas as pd
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- SETUP PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "lat_long_kota_kab.csv")
SECRETS_PATH = os.path.join(BASE_DIR, ".streamlit", "secrets.toml")

# 1. Load API Keys
try:
    secrets = toml.load(SECRETS_PATH)
    GOOGLE_API_KEY = secrets["GOOGLE_API_KEY"]
    QDRANT_URL = secrets["QDRANT_URL"]
    QDRANT_API_KEY = secrets["QDRANT_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except Exception as e:
    print(f"❌ Error membaca secrets.toml: {e}")
    exit()

print("🚀 Memulai Ingestion Lokasi ke Qdrant (Mode Stabil)...")

# 2. Baca CSV
try:
    df = pd.read_csv(CSV_PATH)
    print(f"📄 Berhasil membaca CSV. Total: {len(df)} lokasi.")
except Exception as e:
    print(f"❌ Gagal membaca CSV: {e}")
    exit()

# 3. Konversi ke Documents
documents = []
for index, row in df.iterrows():
    city_name = str(row['name']).strip()
    doc = Document(
        page_content=city_name, 
        metadata={
            "lat": float(row['lat']),
            "long": float(row['long']),
            "original_name": city_name
        }
    )
    documents.append(doc)

print(f"📦 Siap meng-upload {len(documents)} lokasi.")

# 4. Setup Qdrant dengan TIMEOUT TINGGI
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# PENTING: Tambahkan timeout=60 (detik) agar tidak gampang putus
client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
    timeout=60 
)

collection_name = "pikopi_locations" 

# Reset Collection
try:
    client.delete_collection(collection_name)
    print(f"🗑️  Collection '{collection_name}' lama dihapus.")
except:
    pass

print(f"🆕 Membuat collection baru: '{collection_name}'")
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
)

# 5. Upload dengan RETRY LOGIC & BATCH KECIL
batch_size = 20 # Turunkan dari 50 ke 20 biar ringan
total_docs = len(documents)

print("\n⏳ Mulai upload...")

for i in range(0, total_docs, batch_size):
    batch = documents[i : i + batch_size]
    batch_num = (i // batch_size) + 1
    total_batches = (total_docs + batch_size - 1) // batch_size
    
    success = False
    retries = 0
    max_retries = 5 # Coba sampai 5 kali
    
    while not success and retries < max_retries:
        try:
            if retries == 0:
                print(f"   📤 Batch {batch_num}/{total_batches}...", end=" ")
            else:
                print(f"      🔄 Retry {retries}...", end=" ")

            # Upload
            QdrantVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                collection_name=collection_name,
                prefer_grpc=False,
                force_recreate=False 
            )
            
            print("✅ OK")
            success = True
            time.sleep(1) # Istirahat sebentar
            
        except Exception as e:
            print(f"\n      ⚠️ Error: {e}. Tunggu 5 detik...")
            time.sleep(5)
            retries += 1

    if not success:
        print(f"\n❌ Gagal total di Batch {batch_num}. Lewati.")

print("\n🎉 SUKSES! Semua data lokasi sudah masuk Qdrant.")