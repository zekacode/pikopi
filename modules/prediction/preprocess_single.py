# modules/prediction/preprocess_single.py
import pandas as pd
import numpy as np
from dateutil import parser

def preprocess_single(sample: dict) -> pd.DataFrame:

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
    # Gesha
    "Gesha": "Gesha",
    "Typica Gesha": "Gesha",
    "Sl34+Gesha": "Gesha",

    # Caturra
    "Caturra": "Caturra",
    "Red Bourbon,Caturra": "Caturra",
    "Castillo,Caturra,Bourbon": "Caturra",
    "Caturra-Catuai": "Caturra",
    "BOURBON, CATURRA Y CATIMOR": "Caturra",

    # Typica
    "Typica": "Typica",
    "Maragogype": "Typica",
    "Java": "Typica",
    "Gayo": "Typica",
    "Typica + SL34": "Typica",

    # Bourbon
    "Bourbon": "Bourbon",
    "Red Bourbon": "Bourbon",
    "Yellow Bourbon": "Bourbon",
    "Pacas": "Bourbon",
    "Mundo Novo": "Bourbon",
    "Bourbon Sidra": "Bourbon",
    "Catuai and Mundo Novo": "Bourbon",
    "Pacamara": "Bourbon",
    "SL14": "Bourbon",

    # Catuai
    "Catuai": "Catuai",
    "Yellow Catuai": "Catuai",
    "Catucai": "Catuai",
    "Catrenic": "Catuai",
    "MARSELLESA, CATUAI, CATURRA & MARSELLESA, ANACAFE 14, CATUAI": "Catuai",

    # Catimor / Sarchimor
    "Catimor": "Catimor",
    "Castillo": "Catimor",
    "Castillo Paraguaycito": "Catimor",
    "Parainema": "Catimor",
    "Sarchimor": "Catimor",
    "Lempira": "Catimor",
    "Jember,TIM-TIM,Ateng": "Catimor",
    "Castillo and Colombia blend": "Catimor",
    "Catimor,Catuai,Caturra,Bourbon": "Catimor",
    "Typica Bourbon Caturra Catimor": "Catimor",
    "Bourbon, Catimor, Caturra, Typica": "Catimor",
    "Caturra,Colombia,Castillo": "Catimor",

    # Ethiopian Landrace
    "Ethiopian Heirlooms": "Ethiopian Heirlooms",
    "Wolishalo,Kurume,Dega": "Ethiopian Heirlooms",
    "SL28": "Ethiopian Heirlooms",
    "SL28,SL34,Ruiru11": "Ethiopian Heirlooms",

    # SL34 standalone
    "SL34": "SL34",

    # Non varietal / noise
    "SHG": "Other",
    "unknown": "Other",
    "unknow": "Other",
     np.nan: "Other",
    "Santander": "Other"  # region, bukan varietas
    }

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
    numeric_cols = [
        'Altitude','Coffee Age','Moisture Percentage',
        'Category One Defects','Category Two Defects','Quakers',
        'Uniformity','Overall','Flavor','Aftertaste',
        'Balance','Acidity','Aroma','Body'
    ]

    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0

    for c in ['Processing Method', 'Variety']:
        if c not in df.columns:
            df[c] = 'unknown'

    return df
