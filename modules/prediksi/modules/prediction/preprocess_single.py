import pandas as pd
import numpy as np
from dateutil import parser

def preprocess_single(sample: dict, mode: str = "fisik") -> pd.DataFrame:
    """
    Preprocess single sample dict into DataFrame ready for prediction.

    Args:
        sample (dict): dictionary with input features
        mode (str): "fisik" or "akurat"

    Returns:
        pd.DataFrame: preprocessed sample
    """
    df = pd.DataFrame([sample])

    # ----------------------------
    # Processing Method mapping
    # ----------------------------
    processing_mapping = {
        "Double Anaerobic Washed": "Washed / Wet",
        "Semi Washed": "Washed / Wet",
        "Honey,Mossto": "Pulped natural / honey",
        "Double Carbonic Maceration / Natural": "Natural / Dry",
        "Wet Hulling": "Washed / Wet",
        "Anaerobico 1000h": "Washed / Wet",
        "SEMI-LAVADO": "Natural / Dry",
        np.nan: "Washed / Wet"
    }
    if 'Processing Method' in df.columns:
        df['Processing Method'] = df['Processing Method'].replace(processing_mapping)

    # ----------------------------
    # Variety mapping
    # ----------------------------
    variety_mapping = {
        "Santander": "Other",
        "Typica Gesha": "Other",
        "Catucai": "Catuai",
        "Yellow Catuai": "Catuai",
        "unknow": "unknown",
        np.nan: "Other"
    }
    if 'Variety' in df.columns:
        df['Variety'] = df['Variety'].replace(variety_mapping)

    # ----------------------------
    # Clean Altitude
    # ----------------------------
    def clean_altitude(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, str):
            v = val.replace(" ", "").replace("masl", "").replace("m", "")
            if '-' in v:
                try:
                    s, e = v.split('-')
                    return (float(s) + float(e)) / 2
                except:
                    return np.nan
            try:
                return float(v)
            except:
                return np.nan
        try:
            return float(val)
        except:
            return np.nan

    if 'Altitude' in df.columns:
        df['Altitude'] = df['Altitude'].apply(clean_altitude)

    # ----------------------------
    # Coffee Age
    # ----------------------------
    if 'Coffee Age' not in df.columns:
        if 'Harvest Year' in df.columns and 'Expiration' in df.columns:
            try:
                df['Harvest Year'] = pd.to_datetime(
                    df['Harvest Year'].astype(str).str.split('/').str[0].str.strip(),
                    errors='coerce',
                    format='%Y'
                )
                df['Expiration'] = df['Expiration'].apply(
                    lambda x: parser.parse(str(x)) if pd.notnull(x) else pd.NaT
                )
                df['Coffee Age'] = (df['Expiration'] - df['Harvest Year']).dt.days
            except:
                df['Coffee Age'] = 0
        else:
            df['Coffee Age'] = 0

    # ----------------------------
    # Fill NaN numeric
    # ----------------------------
    for c in df.select_dtypes(include=['float64', 'int64']).columns:
        df[c] = df[c].fillna(0)

    # Fill NaN object
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].fillna('unknown')

    # ----------------------------
    # Tambahkan semua kolom numerik & kategorikal sesuai mode
    # ----------------------------
    if mode == "fisik":
        numeric_cols = ['Altitude','Coffee Age','Moisture Percentage',
                        'Category One Defects','Category Two Defects','Quakers']
    else:  # mode == "akurat"
        numeric_cols = ['Uniformity','Clean Cup','Sweetness','Overall','Flavor',
                        'Aftertaste','Balance','Acidity','Aroma','Body']

    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Pastikan kategorikal ada
    for c in ['Processing Method','Variety']:
        if c not in df.columns:
            df[c] = 'unknown'

    return df
