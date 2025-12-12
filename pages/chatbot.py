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
Version: 2.2 (Fixed State Leakage)
"""

import streamlit as st
import sys
import os
import toml
import time
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- PATH CONFIGURATION (CRITICAL FIX) ---
# Ensures Python can locate the 'modules' package even when running from the 'pages' directory.
# This is essential for Streamlit Cloud deployment.

# Get the directory of the current file (pages/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate one level up to the project root (pikopi/)
root_dir = os.path.dirname(current_dir)

# Add root directory to sys.path if not already present
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import tools from the custom module
try:
    from modules.chatbot.tools import tools_list
except ImportError as e:
    st.error(f"Failed to import tools: {e}. Ensure the 'modules' folder contains an __init__.py file.")
    st.stop()

# --- 1. LOGGING CONFIGURATION ---
# Configures the logging system to capture runtime events and errors.
# Logs are saved to 'pikopi.log' with UTF-8 encoding to support emojis.
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
# Injects custom CSS to improve the UI/UX (Dark Mode Support).
st.markdown("""
<style>
    /* User Chat Bubble - Medium Brown */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #3E322C; 
        color: #E6D7C3;
    }
    
    /* Assistant Chat Bubble - Transparent/Default Dark */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: transparent;
    }

    /* Header Title Color - Primary Theme Color */
    h1 {
        color: #D4A373 !important;
    }
    
    /* Google Maps Link Styling - Gold/Bright for visibility */
    a {
        color: #F4A261 !important;
        font-weight: bold;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
        color: #E9C46A !important;
    }
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
        # Load API Keys securely.
        # We use 'root_dir' to construct the path, ensuring it works regardless of the execution context.
        local_secrets_path = os.path.join(root_dir, ".streamlit", "secrets.toml")
        
        if os.path.exists(local_secrets_path):
            secrets = toml.load(local_secrets_path)
            os.environ["GOOGLE_API_KEY"] = secrets["GOOGLE_API_KEY"]
        else:
            # Fallback to Streamlit Cloud Secrets (Environment Variables)
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

# --- HELPER FUNCTION: INIT HISTORY ---
def init_chatbot_history():
    """Returns the initial chat history with System Prompt."""
    return [
        {"role": "system", "content": system_prompt}, # Inject Persona
        {"role": "assistant", "content": "Halo Kak! Mau ngopi apa hari ini? PIKOPI siap bantu cariin cafe atau resep seduh!"}
    ]

# --- 5. SIDEBAR UI ---
with st.sidebar:
    st.title("PIKOPI ☕")
    st.markdown("Click here to reset the chat and start a new conversation.")
    
    # Reset Button: Clears session state to start a new conversation
    if st.button("🗑️ Reset Chat"):
        # Reset to a specific key 'chatbot_messages' to avoid conflict with other pages
        st.session_state.chatbot_messages = init_chatbot_history()
        logger.info("Chat history reset by user.")
        st.rerun()

# --- 6. MAIN CHAT INTERFACE ---
st.title("Ngopi bareng PIKOPI ☕")

# Initialize Session State for Chat History (Unique Key: chatbot_messages)
if "chatbot_messages" not in st.session_state:
    st.session_state.chatbot_messages = init_chatbot_history()

# Render Chat History
# Skips the 'system' message so the user doesn't see the internal instructions.
for msg in st.session_state.chatbot_messages:
    if msg["role"] == "system": continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 7. USER INPUT HANDLING ---
if prompt := st.chat_input("Ketik pertanyaanmu di sini..."):
    # Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Append to history (Unique Key)
    st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
    logger.info(f"User Input: {prompt}")

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("PIKOPI sedang meracik jawaban..."):
            try:
                # Prepare history for LangChain (Convert to Schema)
                chat_history = []
                for msg in st.session_state.chatbot_messages:
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

                # Save AI response to history (Unique Key)
                st.session_state.chatbot_messages.append({"role": "assistant", "content": final_text})
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
                st.session_state.chatbot_messages.append({"role": "assistant", "content": error_msg})