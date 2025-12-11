import os
import joblib
from typing import Tuple, Any

def load_model_and_preprocessor(mode: str) -> Tuple[Any, Any]:
    """
    Load model & preprocessor dari folder 'models' menggunakan absolute path.
    """
    # PERBAIKAN PATH: Folder models ada di sebelah file ini
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    model_obj, preprocessor_obj = None, None

    if mode == "fisik":
        mname, pname = "model_fisik.pkl", "preprocessor_fisik.pkl"
    else:
        mname, pname = "model_akurat.pkl", "preprocessor_akurat.pkl"

    mpath = os.path.join(base, mname)
    ppath = os.path.join(base, pname)

    # Debug (Bisa dihapus nanti kalau sudah jalan)
    # print("Cek path model:", mpath)

    if os.path.exists(mpath):
        model_obj = joblib.load(mpath)
    else:
        print(f"❌ File model tidak ditemukan: {mpath}")

    if os.path.exists(ppath):
        preprocessor_obj = joblib.load(ppath)
    else:
        print(f"❌ File preprocessor tidak ditemukan: {ppath}")

    return model_obj, preprocessor_obj