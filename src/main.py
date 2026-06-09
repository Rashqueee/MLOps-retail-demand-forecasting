import os
import time
import traceback
import numpy as np
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI(
    title="Tokopedia Retail Demand Forecasting API",
    description="API untuk memprediksi demand harian produk menggunakan XGBoost dan MLflow Model Registry",
    version="1.0"
)

# Konfigurasi MLflow Tracking Server dari Environment Docker Compose
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)

MODEL_NAME = "Retail_Demand_XGBoost"
STAGE = "Production"
model = None

def load_production_model():
    """Fungsi untuk memuat model berstatus Production dari MLflow dengan mekanisme retry"""
    global model
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    
    # Mekanisme retry diperlukan karena saat docker-compose up, 
    # API service bisa jadi siap lebih cepat daripada MLflow server selesai booting.
    for attempt in range(10):
        try:
            print(f"[{attempt+1}/10] Mencoba menarik model '{MODEL_NAME}' dari {TRACKING_URI}...")
            model = mlflow.pyfunc.load_model(model_uri)
            print("-> SUKSES: Model berhasil dimuat ke dalam memori API!")
            return
        except Exception as e:
            print(f"-> GAGAL: Server MLflow belum siap atau model belum terdaftar. Mengulang dalam 5 detik...")
            traceback.print_exc()
            time.sleep(5)
            
    print("-> FATAL: API gagal memuat model setelah beberapa kali percobaan.")

@app.on_event("startup")
def startup_event():
    """Dijalankan otomatis saat kontainer API dinyalakan"""
    load_production_model()

# --- SKEMA INPUT DATA (VALIDASI PYDANTIC) ---
class ProductFeature(BaseModel):
    Year: int
    Month: int
    Day: int
    Hour: int
    DayOfWeek: int
    product_id: int
    product_name: str
    price: int
    stock: int
    rating: float

class PredictionRequest(BaseModel):
    inputs: List[ProductFeature]

# --- ENDPOINT API ---

@app.get("/")
def home():
    return {
        "status": "Online",
        "message": "Welcome to Tokopedia Demand Forecasting API Service!",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_demand(payload: PredictionRequest):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum siap di memori server. Sila coba beberapa saat lagi.")
    
    try:
        # 1. Konversi payload JSON menjadi Pandas DataFrame
        data_json = [item.dict() for item in payload.inputs]
        df_input = pd.DataFrame(data_json)
        
        # 2. Proteksi & Penyesuaian Tipe Data Kategorikal (Wajib selaras dengan train.py)
        df_input['product_id'] = df_input['product_id'].astype('category')
        df_input['product_name'] = df_input['product_name'].astype('category')
        
        # 3. Eksekusi Prediksi
        predictions = model.predict(df_input)
        
        # 4. Pasca-proses hasil prediksi (Menghapus nilai negatif & pembulatan)
        formatted_predictions = []
        for pred in predictions:
            clean_pred = max(0, round(pred))
            formatted_predictions.append(clean_pred)
            
        # 5. Mengembalikan hasil sebagai response JSON
        return {
            "status": "Success",
            "predictions": formatted_predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses inferensi: {str(e)}")

if __name__ == "__main__":
    # Menjalankan uvicorn server secara lokal jika dieksekusi langsung via python src/main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)