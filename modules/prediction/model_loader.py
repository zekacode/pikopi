import os
import joblib

# Ambil lokasi file ini berada (modules/prediction)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Folder models ada di dalam folder yang sama
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_model_and_preprocessor(mode="fisik"):
    """
    Load model dan preprocessor.
    Karena hanya ada 1 model (best_model.pkl), kita pakai itu untuk semua mode.
    """
    try:
        # Nama file sesuai screenshot Anda
        model_file = "best_model.pkl"
        prep_file = "preprocessor.pkl"
            
        # Gabungkan path
        model_path = os.path.join(MODELS_DIR, model_file)
        prep_path = os.path.join(MODELS_DIR, prep_file)
            
        # Cek keberadaan file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file tidak ditemukan: {model_path}")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Preprocessor file tidak ditemukan: {prep_path}")

        # Load file .pkl
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        
        return model, preprocessor
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None