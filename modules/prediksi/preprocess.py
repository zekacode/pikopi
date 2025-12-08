import pandas as pd
import numpy as np
from dateutil import parser

def preprocess_batch(df):
    """
    Fungsi untuk preprocessing data kopi
    - Mapping kategori Processing Method dan Variety
    - Fix beberapa nilai Altitude
    - Hitung Coffee Age
    - Drop kolom tidak perlu
    - Fill NaN numerik
    """
    df = df.copy()
    
    # Mapping categorical
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
    df['Processing Method'] = df['Processing Method'].replace(processing_mapping)

    variety_mapping = {
        "Santander" : "Other",
        "Typica Gesha" : "Other",
        "Catucai" : "Catuai",
        "Yellow Catuai" : "Catuai",
        "SL28,SL34,Ruiru11" : "Blend",
        "Caturra-Catuai" : "Blend",
        "Typica Bourbon Caturra Catimor" : "Blend",
        "Caturra,Colombia,Castillo" : "Blend",
        "Castillo,Caturra,Bourbon" : "Blend",
        "unknow" : "unknown",
        "Bourbon, Catimor, Caturra, Typica" : "Blend",
        "Pacas" : "Other",
        "Gayo" : "Other",
        "Castillo" : "Other",
        "Lempira" : "Other",
        np.nan: "Other"
    }
    df['Variety'] = df['Variety'].replace(variety_mapping)

    # Fix Altitude
    df.loc[df['ID'] == 99, 'Altitude'] = 5273
    df.loc[df['ID'] == 105, 'Altitude'] = 1800
    df.loc[df['ID'] == 180, 'Altitude'] = 1400

    def clean_altitude(val):
        if isinstance(val, str):
            val = val.replace(" ", "")
            if '-' in val:
                try:
                    start,end = val.split('-')
                    return (int(start)+int(end))/2
                except: return np.nan
            else:
                try: return int(val)
                except: return np.nan
        return val
    df['Altitude'] = df['Altitude'].apply(clean_altitude)

    # Coffee Age
    df['Harvest Year'] = pd.to_datetime(df['Harvest Year'].astype(str).str.split('/').str[0].str.strip(), errors='coerce', format='%Y')
    df['Expiration'] = df['Expiration'].apply(lambda x: parser.parse(str(x)) if pd.notnull(x) else pd.NaT)
    df['Coffee Age'] = (df['Expiration'] - df['Harvest Year']).dt.days

    # Drop kolom tidak perlu
    drop_cols = ['Unnamed: 0', 'ID', 'Country of Origin','Lot Number', 'Mill', 'ICO Number',
                 'Company', 'Producer', 'Number of Bags', 'Bag Weight' , 'In-Country Partner',
                 'Grading Date', 'Owner','Status','Farm Name','Country','Region','Color',
                 'Harvest Year', 'Expiration', 'Certification Body', 'Certification Address', 'Certification Contact']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Fill NaN numerik dengan 0
    df = df.fillna(0)
    
    return df
