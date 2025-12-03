"""
PIKOPI - AI Agent Tools Module
------------------------------
This module defines the specific capabilities (tools) available to the PIKOPI Agent.
It handles interactions with external APIs (Qdrant, Google Maps, Gemini) and 
performs deterministic calculations.

Tools included:
1.  `retrieve_coffee_knowledge`: RAG-based retrieval for theoretical questions.
2.  `find_cafes_with_maps`: Location-based search using Vector Search + Google Maps Grounding.
3.  `calculate_brew_recipe`: Mathematical logic for coffee brewing ratios.

Author: Putrawin Adha Muzakki
"""

import os
import toml
import pandas as pd
import traceback
import urllib.parse
import logging
from datetime import datetime
import pytz 
import json

# AI & Vector DB Libraries
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.tools import tool

# Google GenAI SDK (for Maps Grounding)
from google import genai
from google.genai import types

# --- 1. LOGGING SETUP ---
# We use the same logging configuration as app.py to centralize debug info.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pikopi.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("🔧 Initializing tools module...")

# --- 2. CONFIGURATION & SECRETS ---
# Load API keys securely. Supports both local development (.streamlit/secrets.toml)
# and cloud deployment environments.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_PATH = os.path.join(BASE_DIR, ".streamlit", "secrets.toml")

GOOGLE_API_KEY = None
QDRANT_URL = None
QDRANT_API_KEY = None

try:
    secrets = toml.load(SECRETS_PATH)
    GOOGLE_API_KEY = secrets["GOOGLE_API_KEY"]
    QDRANT_URL = secrets["QDRANT_URL"]
    QDRANT_API_KEY = secrets["QDRANT_API_KEY"]
    
    # Set env var for LangChain components that require it implicitly
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    logger.info("✅ Secrets loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading secrets: {e}")

# Path to corrections.json for typo handling in location names
CORRECTIONS_PATH = os.path.join(BASE_DIR, "knowledge_base", "corrections.json")

#Helper Function to Load Corrections
def load_corrections():
    """Membaca data koreksi dari JSON agar tidak hardcoded di script"""
    try:
        with open(CORRECTIONS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Gagal memuat corrections.json: {e}")
        return {}

#Load corrections data globally
CORRECTION_DATA = load_corrections()

# --- 3. GLOBAL CLIENT INITIALIZATION ---
# Initialize clients once to avoid overhead on every tool execution.
# Using 'text-embedding-004' for high-quality semantic vectorization.
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- 4. TOOL DEFINITIONS ---

@tool
def retrieve_coffee_knowledge(query: str) -> str:
    """
    Gunakan alat ini untuk menjawab pertanyaan TEORI tentang kopi.
    """
    logger.info(f"TOOL_USE: retrieve_coffee_knowledge | Query: {query}")
    
    # --- 1. CONTEXT INJECTION (DARI JSON) ---
    injected_context = ""
    search_query = query
    
    # Cek setiap kata kunci di JSON
    # Load corrections
    for key, correction_text in CORRECTION_DATA.items():
        if key in query.lower():
            injected_context += f"\n[FAKTA PENTING: {correction_text}]\n"
            if "metode" in query.lower():
                search_query += " method technique brewing"
    
    try:
        vector_store = QdrantVectorStore(client=client, collection_name="pikopi_knowledge", embedding=embeddings)
        
        # Threshold 0.55
        results = vector_store.similarity_search_with_score(search_query, k=3)
        
        valid_docs = []
        for doc, score in results:
            if score > 0.50: 
                valid_docs.append(doc.page_content)
        
        # --- LOGIKA PERBAIKAN ---
        if not valid_docs:
            # Jika Qdrant kosong TAPI ada info di JSON, tampilkan info JSON-nya
            if injected_context:
                return f"Saya belum memiliki artikel lengkap tentang ini, tapi berikut fakta pentingnya:\n{injected_context}"
            else:
                logger.warning(f"Knowledge not found for: {query}")
                return "Maaf, informasi tersebut tidak ditemukan dalam database pengetahuan kopi saya."
            
        context = injected_context + "\n\n" + "\n\n".join(valid_docs)
        return context
        
    except Exception as e:
        logger.error(f"Error knowledge tool: {e}")
        return f"Error knowledge: {e}"

@tool
def find_cafes_with_maps(city_name: str, preferences: str = "") -> str:
    """
    Use this tool when the user asks for CAFE RECOMMENDATIONS in a specific CITY.
    
    Args:
        city_name: The name of the city (e.g., "Surabaya", "Jakarta Selatan").
        preferences: Specific user needs (e.g., "outdoor seating", "open 24 hours").
    """
    logger.info(f"TOOL_USE: find_cafes_with_maps | City: {city_name} | Pref: {preferences}")
    
    try:
        # --- STEP 1: Resolve Location using Vector Search ---
        # We use Qdrant instead of exact string matching (Pandas) to handle typos.
        # e.g., "Surbaya" -> matches "KOTA SURABAYA" vector.
        vector_store_loc = QdrantVectorStore(
            client=client, 
            collection_name="pikopi_locations", 
            embedding=embeddings
        )
        results = vector_store_loc.similarity_search_with_score(city_name, k=1)
        
        # Strict threshold (0.75) to ensure we don't map random words to cities.
        if not results or results[0][1] < 0.50:
            logger.warning(f"City not found or low score: {city_name}")
            return f"Waduh, PIKOPI belum punya data lokasi untuk '{city_name}', atau mungkin ada typo? Coba ketik nama kotanya yang lengkap ya Kak (misal: 'Kota Surabaya')."
            
        found_city = results[0][0]
        found_name = found_city.page_content
        latitude = found_city.metadata.get("lat")
        longitude = found_city.metadata.get("long")
        
        logger.info(f"📍 Location Resolved: {found_name} ({latitude}, {longitude})")
        
        # --- STEP 2: Context-Aware Prompting (Time) ---
        # Get current time in Indonesia (WIB) to recommend places that are OPEN NOW.
        tz = pytz.timezone('Asia/Jakarta')
        now = datetime.now(tz)
        current_time = now.strftime("%H:%M")
        day_name = now.strftime("%A")
        
        client_genai = genai.Client(api_key=GOOGLE_API_KEY)
        
        prompt_context = f"Saat ini hari {day_name}, jam {current_time} WIB."
        
        if preferences:
            prompt = (
                f"{prompt_context} Carikan rekomendasi coffee shop di {found_name} yang BUKA JAM SEGINI "
                f"dan memiliki kriteria: {preferences}. \n"
                f"Untuk setiap tempat, jelaskan secara singkat:\n"
                f"1. Kenapa tempat ini cocok dengan kriteria user?\n"
                f"2. Apa menu andalan (signature dish) mereka?\n"
                f"3. Bagaimana vibe/suasananya (misal: tenang, ramai, instagramable)?"
                f"4. **Price Level**: (Cari tanda $, $$, atau $$$ di data Google Maps. Jika tidak ada, tulis 'Info harga tidak tersedia').\n"
            )
        else:
            prompt = (
                f"{prompt_context} Carikan 5 rekomendasi coffee shop terbaik/populer di {found_name} yang BUKA JAM SEGINI.\n"
                f"Berikan ulasan menarik yang mencakup:\n"
                f"- Rating & Popularitas.\n"
                f"- Vibe tempat (cocok buat kerja atau nongkrong).\n"
                f"- **Price Level**: (Cari tanda $, $$, atau $$$ di data Google Maps. Jika tidak ada, tulis 'Info harga tidak tersedia').\n"
            )

        # --- STEP 3: Call Gemini with Google Maps Grounding ---
        # This connects the LLM to real-world Maps data.
        response = client_genai.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_maps=types.GoogleMaps())],
                tool_config=types.ToolConfig(
                    retrieval_config=types.RetrievalConfig(
                        lat_lng=types.LatLng(latitude=latitude, longitude=longitude)
                    )
                ),
            ),
        )

        # --- STEP 4: Format Output & Generate Fallback Links ---
        final_output = response.text + "\n\n"
        
        links_added = False
        # Check if Gemini returned grounding metadata (source links)
        if response.candidates and response.candidates[0].grounding_metadata:
            grounding = response.candidates[0].grounding_metadata
            if grounding.grounding_chunks:
                final_output += "---\n**📍 Link Google Maps:**\n"
                for chunk in grounding.grounding_chunks:
                    if chunk.maps:
                        title = chunk.maps.title
                        
                        # Robust URI Extraction:
                        # Sometimes the API returns 'google_maps_uri_variant', sometimes 'uri'.
                        # We use getattr to avoid AttributeError crashes.
                        uri = getattr(chunk.maps, 'google_maps_uri_variant', None) or getattr(chunk.maps, 'uri', None)
                        
                        # Fallback Mechanism:
                        # If API returns no link, we generate a manual Search URL.
                        if not uri:
                            safe_title = urllib.parse.quote(title)
                            uri = f"https://www.google.com/maps/search/?api=1&query={safe_title}"
                        
                        final_output += f"1. [{title}]({uri})\n"
                        links_added = True
        
        # Global Fallback: If no specific links were found, provide a general search link.
        if not links_added:
            safe_city = urllib.parse.quote(found_name)
            final_output += f"\n\n[👉 Cari Cafe di {found_name} via Google Maps](https://www.google.com/maps/search/coffee+shop+in+{safe_city})"

        return final_output

    except Exception as e:
        logger.error(f"Error in Maps Tool: {e}", exc_info=True)
        return f"Gagal mencari lokasi: {e}"

@tool
def calculate_brew_recipe(coffee_grams: float = 0, water_ml: float = 0, method: str = "v60", strength: str = "balanced") -> str:
    """
    Use this tool to CALCULATE coffee brewing recipes mathematically.
    Do not use this for searching information, only for calculation.
    
    Args:
        coffee_grams: Amount of coffee beans (grams).
        water_ml: Target water amount (ml).
        method: Brewing method (v60, french press, espresso, tubruk, etc).
        strength: Desired taste ('strong', 'balanced', 'light').
    """
    logger.info(f"TOOL_USE: calculate_brew_recipe | Method: {method}")
    
    # --- LOGIC: Standard Ratios Dictionary ---
    # We use a static dictionary instead of a database because brewing ratios are standard rules.
    brew_standards = {
        "v60": {"strong": 13, "balanced": 15, "light": 17},
        "kalita": {"strong": 13, "balanced": 15, "light": 17},
        "chemex": {"strong": 14, "balanced": 16, "light": 18},
        "french press": {"strong": 10, "balanced": 12, "light": 14},
        "tubruk": {"strong": 10, "balanced": 12, "light": 14},
        "aeropress": {"strong": 10, "balanced": 14, "light": 16},
        "cold brew": {"strong": 8, "balanced": 10, "light": 12},
        "espresso": {"strong": 1.5, "balanced": 2, "light": 2.5},
    }
    
    # Normalize input
    m = method.lower()
    selected_method = "v60" # Default fallback
    for key in brew_standards:
        if key in m:
            selected_method = key
            break
            
    standards = brew_standards[selected_method]
    s = strength.lower()
    
    # Determine Ratio based on Strength
    if "strong" in s or "pekat" in s: 
        ratio = standards["strong"]
        desc = "Pekat"
    elif "light" in s or "encer" in s: 
        ratio = standards["light"]
        desc = "Ringan"
    else: 
        ratio = standards["balanced"]
        desc = "Seimbang"
        
    # Calculate
    result_text = ""
    if coffee_grams > 0:
        water_needed = coffee_grams * ratio
        result_text = f"### 🧮 Resep {selected_method.title()} ({desc})\n**Rasio 1:{ratio}**\n\n- Bubuk Kopi: **{coffee_grams}g**\n- Air: **{water_needed:.0f}ml**"
    elif water_ml > 0:
        coffee_needed = water_ml / ratio
        result_text = f"### 🧮 Resep {selected_method.title()} ({desc})\n**Rasio 1:{ratio}**\n\n- Air: **{water_ml}ml**\n- Bubuk Kopi: **{coffee_needed:.1f}g**"
    else:
        return "Sebutkan jumlah kopi atau air."

    # Add Method-Specific Tips
    tips = {
        "v60": "Gilingan: Medium-Fine. Tuang memutar.",
        "french press": "Gilingan: Coarse. Rendam 4 menit.",
        "espresso": "Gilingan: Fine. Ekstraksi 25-30 detik.",
        "tubruk": "Gilingan: Medium. Air mendidih, aduk, diamkan 4 menit."
    }
    result_text += f"\n\n**Tips:** {tips.get(selected_method, tips['v60'])}"
    return result_text

# Export tools list for the Agent
tools_list = [retrieve_coffee_knowledge, find_cafes_with_maps, calculate_brew_recipe]