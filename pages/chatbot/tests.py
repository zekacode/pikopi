import pytest
import json
import os
from tools import calculate_brew_recipe, retrieve_coffee_knowledge, find_cafes_with_maps

# Load data koreksi untuk verifikasi tes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORRECTIONS_PATH = os.path.join(BASE_DIR, "knowledge_base", "corrections.json")

with open(CORRECTIONS_PATH, 'r', encoding='utf-8') as f:
    CORRECTIONS_DATA = json.load(f)

# ==============================================================================
# GROUP 1: SMART BREW CALCULATOR (32 TEST CASES)
# Menguji kombinasi Metode x Strength x Input (Kopi/Air)
# ==============================================================================

# Data Test: (Method, Strength, Input Grams, Input Water, Expected Output Keyword)
CALCULATOR_SCENARIOS = [
    # --- V60 (Standard 1:15, Strong 1:13, Light 1:17) ---
    ("v60", "balanced", 15, 0, "225"),      # 15 * 15 = 225
    ("v60", "strong", 15, 0, "195"),        # 15 * 13 = 195
    ("v60", "light", 15, 0, "255"),         # 15 * 17 = 255
    ("v60", "balanced", 0, 300, "20.0"),    # 300 / 15 = 20
    
    # --- French Press (Standard 1:12, Strong 1:10) ---
    ("french press", "balanced", 20, 0, "240"), # 20 * 12 = 240
    ("french press", "strong", 20, 0, "200"),   # 20 * 10 = 200
    ("french press", "light", 20, 0, "280"),    # 20 * 14 = 280
    
    # --- Espresso (Standard 1:2, Strong 1:1.5) ---
    ("espresso", "balanced", 18, 0, "36"),      # 18 * 2 = 36
    ("espresso", "strong", 18, 0, "27"),        # 18 * 1.5 = 27
    ("espresso", "light", 18, 0, "45"),         # 18 * 2.5 = 45
    
    # --- Aeropress (Standard 1:14) ---
    ("aeropress", "balanced", 15, 0, "210"),    # 15 * 14 = 210
    
    # --- Cold Brew (Konsentrat 1:10) ---
    ("cold brew", "balanced", 50, 0, "500"),    # 50 * 10 = 500
    ("cold brew", "strong", 50, 0, "400"),      # 50 * 8 = 400
    
    # --- Tubruk (Standard 1:12) ---
    ("tubruk", "balanced", 10, 0, "120"),       # 10 * 12 = 120
    
    # --- Kalita (Sama kayak V60) ---
    ("kalita", "balanced", 15, 0, "225"),
    
    # --- Chemex (Agak light 1:16) ---
    ("chemex", "balanced", 20, 0, "320"),       # 20 * 16 = 320
]

@pytest.mark.parametrize("method, strength, grams, water, expected", CALCULATOR_SCENARIOS)
def test_calculator_logic(method, strength, grams, water, expected):
    """Menguji logika matematika untuk berbagai metode seduh"""
    result = calculate_brew_recipe.invoke({
        "coffee_grams": grams, 
        "water_ml": water, 
        "method": method, 
        "strength": strength
    })
    # Cek apakah angka hasil perhitungan ada di output
    assert expected in result
    # Cek apakah nama metode muncul di judul resep
    assert method.title() in result or method.upper() in result

# ==============================================================================
# GROUP 2: CONTEXT INJECTION / SEMANTIC CORRECTION (17 TEST CASES)
# Menguji apakah AI meluruskan pemahaman user tentang Alat vs Metode
# ==============================================================================

# Kita ambil semua key dari corrections.json untuk dites satu per satu
INJECTION_SCENARIOS = [(key) for key in CORRECTIONS_DATA.keys()]

@pytest.mark.parametrize("keyword", INJECTION_SCENARIOS)
def test_context_injection(keyword):
    """Menguji apakah tool menyuntikkan fakta koreksi dari JSON"""
    query = f"Jelaskan metode {keyword}"
    result = retrieve_coffee_knowledge.invoke(query)
    
    # Ekspektasi: Output harus mengandung kata "FAKTA PENTING"
    assert "FAKTA PENTING" in result
    
    # Cek keyword (handle kasus 'mokapot' vs 'moka pot')
    # Kita split keyword jadi kata per kata, minimal satu kata harus muncul
    keyword_parts = keyword.split() 
    assert any(part in result.lower() for part in keyword_parts)

# ==============================================================================
# GROUP 3: RAG KNOWLEDGE RETRIEVAL (5 TEST CASES)
# Menguji pengambilan fakta umum dari Qdrant
# ==============================================================================

RAG_SCENARIOS = [
    ("Jelaskan karakteristik biji kopi Arabika", ["arabica", "arabika", "acid"]), # Tambah 'arabica'
    ("Jelaskan rasa Robusta", ["robusta", "bitter", "body"]),
    ("Sejarah kopi", ["history", "sejarah", "dutch", "belanda"]), # Tambah 'history'
    ("Apa itu Green Bean?", ["green", "bean", "mentah"]),
    ("Proses pasca panen", ["process", "wash", "natural", "honey"]), # Tambah 'process'
]

@pytest.mark.parametrize("query, keywords", RAG_SCENARIOS)
def test_rag_retrieval(query, keywords):
    """Menguji apakah RAG mengembalikan dokumen yang relevan"""
    result = retrieve_coffee_knowledge.invoke(query)
    # Cek minimal satu keyword muncul
    assert any(k in result.lower() for k in keywords)

# ==============================================================================
# GROUP 4: MAPS & LOCATION ROBUSTNESS (6 TEST CASES)
# Menguji ketahanan pencarian lokasi (Typo, Kota Kecil, dll)
# ==============================================================================

MAPS_SCENARIOS = [
    ("Kota Surabaya", True),   # Exact match
    ("Surbaya", True),         # Typo (Vector Search harus bisa handle ini)
    ("Malang", True),          # Partial match
    ("Jakarta Selatan", True), # Specific region
    ("Wakanda Forever", False),# Fictional (Harus Fail/Warning)
    ("Konoha", False)          # Fictional
]

@pytest.mark.parametrize("city, should_find", MAPS_SCENARIOS)
def test_maps_location_search(city, should_find):
    """Menguji apakah sistem bisa menemukan (atau menolak) lokasi"""
    result = find_cafes_with_maps.invoke({
        "city_name": city,
        "preferences": ""
    })
    
    if should_find:
        # Jika harus ketemu, outputnya adalah rekomendasi
        assert "rekomendasi" in result.lower() or "link google maps" in result.lower()
    else:
        # Jika tidak boleh ketemu, outputnya adalah pesan error ramah
        assert "waduh" in result.lower() or "belum punya data" in result.lower()

# ==============================================================================
# GROUP 5: EDGE CASES & ERROR HANDLING (5 TEST CASES)
# Menguji input aneh-aneh
# ==============================================================================

def test_calc_zero_input():
    """Input 0 gram dan 0 air"""
    result = calculate_brew_recipe.invoke({
        "coffee_grams": 0, "water_ml": 0, "method": "v60"
    })
    assert "Sebutkan jumlah" in result

def test_calc_negative_input():
    """Input negatif (Logic check)"""
    # Di tools.py logicnya: if coffee_grams > 0. Jadi negatif dianggap tidak valid.
    result = calculate_brew_recipe.invoke({
        "coffee_grams": -5, "water_ml": 0, "method": "v60"
    })
    assert "Sebutkan jumlah" in result

def test_rag_irrelevant_topic():
    """Topik di luar kopi (Threshold check)"""
    result = retrieve_coffee_knowledge.invoke("Cara merakit bom nuklir")
    assert "maaf" in result.lower() or "tidak ditemukan" in result.lower()

def test_maps_empty_city():
    """Nama kota kosong"""
    # Ini akan mentrigger error handling di Qdrant search
    result = find_cafes_with_maps.invoke({"city_name": "", "preferences": ""})
    assert "maaf" in result.lower() or "waduh" in result.lower() or "error" in result.lower()

def test_maps_price_prompt_injection():
    """Cek apakah prompt harga masuk"""
    # Kita tidak bisa cek output Gemini pasti, tapi kita bisa cek apakah fungsi jalan
    result = find_cafes_with_maps.invoke({"city_name": "Kota Bandung", "preferences": "murah"})
    assert "rekomendasi" in result.lower()