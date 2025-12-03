import os
import time
import toml
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- SETUP PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_base")
SECRETS_PATH = os.path.join(BASE_DIR, ".streamlit", "secrets.toml")

# Load API Keys
try:
    secrets = toml.load(SECRETS_PATH)
    GOOGLE_API_KEY = secrets["GOOGLE_API_KEY"]
    QDRANT_URL = secrets["QDRANT_URL"]
    QDRANT_API_KEY = secrets["QDRANT_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except Exception as e:
    print(f"❌ Error membaca secrets.toml: {e}")
    exit()

print("🚀 Memulai proses ingestion ...")

# Memuat Dokumen
documents = []
if not os.path.exists(KNOWLEDGE_DIR):
    print(f"❌ Folder {KNOWLEDGE_DIR} tidak ditemukan!")
    exit()

for root, dirs, files in os.walk(KNOWLEDGE_DIR):
    for filename in files:
        if filename.lower().endswith((".md", ".txt")):
            file_path = os.path.join(root, filename)
            try:
                loader = TextLoader(file_path, autodetect_encoding=True)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"   ⚠️ Gagal baca {filename}: {e}")

print(f"📦 Total dokumen dimuat: {len(documents)}")

# Memecah Teks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)
print(f"✂️  Dipecah menjadi {len(splits)} chunks.")

# Setup Qdrant & Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
collection_name = "pikopi_knowledge"

# Hapus collection lama jika ada
try:
    client.delete_collection(collection_name)
    print(f"🗑️  Collection lama dihapus (karena ganti model).")
except:
    pass

# Buat collection baru
print(f"🆕 Membuat collection baru: '{collection_name}'")
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
)

# Upload 
batch_size = 10
total_splits = len(splits)

print("\n⏳ Mulai upload ke Qdrant...")

for i in range(0, total_splits, batch_size):
    batch = splits[i : i + batch_size]
    batch_num = (i // batch_size) + 1
    total_batches = (total_splits + batch_size - 1) // batch_size
    
    success = False
    retries = 0
    max_retries = 3 # Coba 3 kali kalau gagal koneksi
    
    while not success and retries < max_retries:
        try:
            if retries == 0:
                print(f"   📤 Batch {batch_num}/{total_batches}...", end=" ")
            else:
                print(f"      🔄 Retry {retries}...", end=" ")

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
            error_msg = str(e)
            if "429" in error_msg or "Quota exceeded" in error_msg:
                print(f"\n      🛑 Limit Google. Tidur 60 detik...")
                time.sleep(60)
            else:
                print(f"\n      ⚠️ Error Koneksi: {e}. Tunggu 5 detik...")
                time.sleep(5)
                retries += 1

    if not success:
        print(f"\n❌ Gagal total di Batch {batch_num} setelah {max_retries} kali coba. Lanjut ke batch berikutnya.")

print("\n🎉 SUKSES! Data berhasil masuk.")