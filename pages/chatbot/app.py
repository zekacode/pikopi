"""
PIKOPI - AI Coffee Expert Application
-------------------------------------
This is the main entry point for the Streamlit application.
It serves as the frontend interface for the PIKOPI Agent.

Key Features:
1.  **Chat Interface:** A WhatsApp-like chat UI for user interaction.
2.  **Agent Orchestration:** Connects to Google Gemini via LangChain/LangGraph.
3.  **State Management:** Handles chat history and session persistence.
4.  **UX Enhancements:** Includes typing effects and custom CSS styling.
5.  **Robust Error Handling:** User-friendly error messages with backend logging.

Author: Putrawin Adha Muzakki
"""

import streamlit as st
import os
import toml
import time
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tools import tools_list

# --- 1. LOGGING CONFIGURATION ---
# Configures the logging system to capture runtime events and errors.
# Logs are saved to 'pikopi.log' for debugging and printed to the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pikopi.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 2. STREAMLIT PAGE CONFIGURATION ---
# Sets up the browser tab title, favicon, and layout mode.
st.set_page_config(
    page_title="PIKOPI - AI Coffee Expert",
    page_icon="☕",
    layout="centered"
)

# --- 3. CUSTOM CSS STYLING ---
# Injects custom CSS to improve the UI/UX (rounded chat bubbles, link colors).
st.markdown("""
<style>
    /* Chat bubble styling */
    .stChatMessage { border-radius: 10px; padding: 10px; }
    .stChatMessage[data-testid="stChatMessageUser"] { background-color: #f0f2f6; }
    
    /* Header color */
    h1 { color: #6F4E37; }
    
    /* Hyperlink styling for Google Maps links */
    a { color: #1E90FF; font-weight: bold; text-decoration: none; }
    a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# --- 4. AGENT INITIALIZATION ---
@st.cache_resource
def get_agent():
    """
    Initializes the LangChain Agent with Google Gemini and custom tools.
    Uses @st.cache_resource to load the model only once per session to improve performance.
    
    Returns:
        agent_executor: The compiled LangGraph agent ready for invocation.
    """
    try:
        # Load API Keys from local secrets.toml or Streamlit Cloud secrets
        if os.path.exists(".streamlit/secrets.toml"):
            secrets = toml.load(".streamlit/secrets.toml")
            os.environ["GOOGLE_API_KEY"] = secrets["GOOGLE_API_KEY"]
        else:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        
        logger.info("Agent initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return None

    # Initialize the LLM (Brain)
    # Using gemini-2.5-flash for a balance of speed and reasoning capability.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # Create the ReAct Agent (Reasoning + Acting)
    return create_react_agent(llm, tools_list)

agent_executor = get_agent()

# --- 5. SIDEBAR UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/751/751621.png", width=100)
    st.title("PIKOPI ☕")
    st.markdown("Asisten Barista Pintar")
    
    # Reset Button: Clears session state to start a new conversation
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        logger.info("Chat history reset by user.")
        st.rerun()

# --- 6. MAIN CHAT INTERFACE ---
st.title("Ngopi bareng PIKOPI ☕")

# --- SYSTEM PROMPT & PERSONA ---
# Defines the AI's personality (Friendly Barista) and strict operational rules.
# Crucial for ensuring Google Maps links are preserved in the final output.
system_prompt = """
PERAN:
Kamu adalah PIKOPI, asisten virtual spesialis kopi yang cerdas, ramah, dan "gaul". 
Tugasmu hanya satu: Membantu pengguna menikmati dunia kopi, mulai dari mencari cafe hingga belajar teknik seduh.

GAYA BICARA (TONE):
- Gunakan bahasa Indonesia yang santai, akrab, tapi tetap sopan (gunakan sapaan "Kak").
- Hindari bahasa yang terlalu kaku/baku seperti robot.
- Tunjukkan antusiasme (passion) saat membahas kopi lokal Indonesia.

STRATEGI ENGAGEMENT:
- Jangan biarkan percakapan mati.
- Setiap kali selesai menjawab, BERIKAN PERTANYAAN BALIK yang relevan untuk memancing user ngobrol lagi.

CONTOH INTERAKSI:
User: "Halo"
PIKOPI: "Halo Kak! 👋 Wah, jam segini enak nih buat ngopi. Lagi cari rekomendasi cafe atau mau seduh sendiri di rumah?"

User: "Apa bedanya Arabika sama Robusta?"
PIKOPI: "Oke, simpelnya gini Kak:
☕ **Arabika**: Rasanya lebih asam segar (fruity), wangi, dan kafeinnya lebih rendah. Cocok buat yang suka kopi hitam santai.
☕ **Robusta**: Rasanya pahit nendang, body tebal, kafein tinggi. Ini rajanya kopi susu!
Kakak biasanya lebih suka yang mana nih?"

User: "Cariin cafe di Bandung dong"
PIKOPI: "Siap! Bandung mah surganya kopi. Kakak lagi di daerah mana? Atau mau cari yang suasananya sejuk buat healing?"


BATASAN TOPIK (GUARDRAILS):
- Kamu HANYA boleh menjawab pertanyaan seputar: Kopi, Cafe, Teh, Alat Seduh, dan Industri F&B terkait.
- Jika user bertanya topik lain (misal: Politik, Coding, Matematika umum, Curhat asmara), tolak dengan halus dan bercanda.
  Contoh tolak: "Waduh, kalau soal itu PIKOPI kurang paham Kak. Tapi kalau ditanya soal V60 atau cafe hits, PIKOPI jagonya!"

ATURAN TEKNIS (CRITICAL):
1. Jika menggunakan tool 'find_cafes_with_maps', tool akan memberikan Link Google Maps.
2. WAJIB MENYALIN DAN MENAMPILKAN ULANG semua Link Google Maps tersebut di akhir jawabanmu.
3. Tampilkan link dari tempat yang kamu rekomendasikan saja, jangan semua link yang diberikan tool.
4. JANGAN PERNAH MENGHAPUS LINK. Format: [Nama Tempat](URL).
"""

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}, # Inject Persona
        {"role": "assistant", "content": "Halo Kak! Mau ngopi apa hari ini? PIKOPI siap bantu cariin cafe atau resep seduh!"}
    ]

# Render Chat History
# Skips the 'system' message so the user doesn't see the internal instructions.
for msg in st.session_state.messages:
    if msg["role"] == "system": continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 7. USER INPUT HANDLING ---
if prompt := st.chat_input("Ketik pertanyaanmu di sini..."):
    # Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Append to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    logger.info(f"User Input: {prompt}")

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("PIKOPI sedang meracik jawaban..."):
            try:
                # Prepare history for LangChain (Convert to Schema)
                chat_history = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user": 
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant": 
                        chat_history.append(AIMessage(content=msg["content"]))
                    elif msg["role"] == "system": 
                        chat_history.append(SystemMessage(content=msg["content"]))
                
                # Invoke the Agent
                response = agent_executor.invoke({"messages": chat_history})
                raw_response = response["messages"][-1].content
                
                # Parse Response (Handle potential List/JSON output from Gemini)
                final_text = ""
                if isinstance(raw_response, list):
                    for block in raw_response:
                        if isinstance(block, dict) and "text" in block: 
                            final_text += block["text"]
                        elif isinstance(block, str): 
                            final_text += block
                else:
                    final_text = str(raw_response)

                # --- TYPING EFFECT SIMULATION ---
                # Improves UX by simulating a real-time typing experience
                message_placeholder = st.empty()
                displayed_text = ""
                for char in final_text:
                    displayed_text += char
                    message_placeholder.markdown(displayed_text + "▌") # Cursor
                    time.sleep(0.005) # Typing speed
                message_placeholder.markdown(displayed_text) # Final render without cursor

                # Save AI response to history
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                logger.info("Response generated successfully.")
                
                # Force Rerun to sync state and prevent "ghosting" UI issues
                time.sleep(0.5)
                st.rerun()
                
            except Exception as e:
                # Graceful Error Handling
                # Logs the full error for devs, shows a friendly message to the user.
                logger.error(f"Runtime Error: {e}", exc_info=True)
                
                error_msg = "Waduh, mesin PIKOPI lagi agak ngadat nih Kak (koneksi terputus atau error sistem). Coba tanya lagi pelan-pelan ya, atau refresh halamannya."
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})