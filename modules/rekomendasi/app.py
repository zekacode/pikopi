import streamlit as st
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Barista AI Rekomen",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Asisten Barista AI")
st.write("Ceritain rasa kopi yang kamu suka, nanti aku cariin yang cocok!")

# --- LOAD RESOURCES (Di-cache biar cepat) ---
# Kita pakai @st.cache_resource supaya koneksi ke Qdrant & Google AI 
# hanya dilakukan sekali saat aplikasi pertama dibuka, bukan tiap kali user nanya.
@st.cache_resource
def get_rag_chain():
    # Mengambil API Key dari st.secrets (file secrets.toml)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    COLLECTION_NAME = "coffee_review_taste"

    # 1. Setup Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query" 
    )

    # 2. Setup LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7
    )

    # 3. Setup Qdrant Client & Store
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # 4. Retriever & Chain
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
    
    return rag_chain

# Inisialisasi Chain
try:
    chain = get_rag_chain()
except Exception as e:
    st.error(f"Gagal koneksi ke Database/AI: {e}")
    st.stop()

# --- INTERFACE CHAT ---
# Simpan history chat di session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Kamu lagi pengen kopi yang rasanya gimana? (Misal: fruity, strong, atau creamy?)"}
    ]

# Tampilkan history chat yang ada
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input User
if prompt := st.chat_input("Ketik preferensi kopimu di sini..."):
    # 1. Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Proses jawaban AI
    with st.chat_message("assistant"):
        with st.spinner("Sedang meracik rekomendasi..."):
            try:
                response = chain.invoke(prompt)
                st.markdown(response)
                
                # Simpan jawaban assistant ke history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Maaf, ada error: {e}")