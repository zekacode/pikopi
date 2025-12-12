import os
import joblib
from typing import Tuple, Any

def load_model_and_preprocessor() -> Tuple[Any, Any]:
    # Path folder /models relatif terhadap file ini
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))

    # Nama file model dan preprocessor
    mname = "best_model.pkl"
    pname = "preprocessor.pkl"

    # Full path
    mpath = os.path.join(base, mname)
    ppath = os.path.join(base, pname)

    # Debug path
    print("Cek path model:", mpath)
    print("Cek path preprocessor:", ppath)

    # Load model
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"❌ Model file tidak ditemukan: {mpath}")
    model_obj = joblib.load(mpath)

    # Load preprocessor
    if not os.path.exists(ppath):
        raise FileNotFoundError(f"❌ Preprocessor file tidak ditemukan: {ppath}")
    preprocessor_obj = joblib.load(ppath)

    return model_obj, preprocessor_obj