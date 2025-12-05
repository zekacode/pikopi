import streamlit as st
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="Barista AI Rekomen",
    page_icon="☕",
    layout="centered"
)

def reset_conversation():
    st.session_state.messages = []
    st.session_state.recommendation_done = False

with st.sidebar:
    st.header("Pengaturan")
    st.write("Ingin cari kopi lain?")
    if st.button("🔄 Mulai Percakapan Baru", type="primary"):
        reset_conversation()
        st.rerun()

st.title("☕ Asisten Barista AI")
st.write("Ceritakan rasa kopi yang kamu inginkan, aku akan carikan yang pas!")

@st.cache_resource
def get_rag_chain():
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        QDRANT_URL = st.secrets["QDRANT_URL"]
        QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
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

    except Exception as e:
        st.error(f"Error konfigurasi Database/AI: {e}")
        return None

chain = get_rag_chain()

if not chain:
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "recommendation_done" not in st.session_state:
    st.session_state.recommendation_done = False

with st.container():
    st.subheader("1. Tentukan Profil Rasa")
    
    rasa_options = [
        "Fruity (Buah-buahan)", 
        "Acidity (Asam Segar)", 
        "Manis/Sweet", 
        "Pahit/Bitter", 
        "Strong/Bold", 
        "Nutty (Kacang)", 
        "Chocolate", 
        "Caramel", 
        "Spices (Rempah)", 
        "Floral (Bunga)", 
        "Creamy"
    ]
    selected_rasa = st.multiselect("Pilih karakteristik rasa yang kamu cari:", rasa_options, placeholder="Pilih rasa...")

    prompt_requirements = [] 
    
    if selected_rasa:
        st.markdown("---")
        st.write("🎚️ **Atur Intensitas Rasa:**")
        
        for rasa in selected_rasa:
            clean_rasa_name = rasa.split(" (")[0] 
            
            level = st.slider(f"Seberapa kuat rasa **{clean_rasa_name}**?", 1, 10, 5, key=f"slider_{clean_rasa_name}")
            
            if level <= 3:
                desc = "halus/tipis (hint of)"
            elif level <= 7:
                desc = "sedang/seimbang (balanced)"
            else:
                desc = "sangat kuat/dominan (bold/intense)"
            
            prompt_requirements.append(f"- Rasa {clean_rasa_name}: {desc} (Level {level}/10)")

    if st.button("🔍 Temukan Kopi", type="primary"):
        if not selected_rasa:
            st.warning("Mohon pilih minimal satu karakteristik rasa.")
        else:
            req_text = "\n".join(prompt_requirements)
            
            final_user_query = f"""
            Tolong carikan kopi dengan spesifikasi rasa berikut:
            
            {req_text}
            
            Abaikan asal daerah/jenis biji, fokus hanya pada profil rasa yang diminta di atas.
            Berikan rekomendasi yang paling mendekati kombinasi tersebut.
            """
            
            st.session_state.messages = [] 
            
            display_message = f"Aku cari kopi dengan profil: **{', '.join(selected_rasa)}**."
            st.session_state.messages.append({"role": "user", "content": display_message})
            
            with st.spinner("Barista sedang mencicipi data..."):
                try:
                    response = chain.invoke(final_user_query)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.recommendation_done = True 
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Gagal mencari rekomendasi: {e}")


if st.session_state.recommendation_done:
    st.divider()
    st.subheader("2. Rekomendasi & Diskusi")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Tanya detail lain (misal: cara seduh, harga, dll)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Mengetik..."):
                try:
                    response = chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")

elif not st.session_state.recommendation_done and st.session_state.messages:
    st.session_state.messages = []