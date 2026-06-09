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
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram

app = FastAPI(
    title="Tokopedia Retail Demand Forecasting API",
    description="API untuk memprediksi demand dengan Observability (Prometheus)",
    version="1.1"
)

Instrumentator().instrument(app).expose(app)

# Inisialisasi Metrik Kustom untuk mendeteksi Data Drift (Pergeseran Distribusi Prediksi)
PREDICTION_DISTRIBUTION = Histogram(
    "model_prediction_value",
    "Distribusi nilai prediksi demand untuk deteksi data drift",
    buckets=[0, 1, 2, 3, 5, 10, 20, 50, 100]
)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)

MODEL_NAME = "Retail_Demand_XGBoost"
STAGE = "Production"
model = None

def load_production_model():
    global model
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    for attempt in range(10):
        try:
            print(f"[{attempt+1}/10] Mencoba menarik model '{MODEL_NAME}' dari {TRACKING_URI}...")
            model = mlflow.pyfunc.load_model(model_uri)
            print("-> SUKSES: Model berhasil dimuat ke dalam memori API!")
            return
        except Exception as e:
            # Menampilkan error asli agar mudah di-debug jika gagal lagi
            print(f"-> GAGAL (Alasan: {e}). Mengulang dalam 5 detik...")
            time.sleep(5)
            
    print("-> FATAL: API gagal memuat model setelah beberapa kali percobaan.")

@app.on_event("startup")
def startup_event():
    load_production_model()

class ProductFeature(BaseModel):
    Year: int
    Month: int
    Day: int
    Hour: int
    DayOfWeek: int
    product_id: int
    product_name: int
    price: int
    stock: int
    rating: float

class PredictionRequest(BaseModel):
    inputs: List[ProductFeature]

@app.get("/")
def home():
    return {"status": "Online", "model_loaded": model is not None}

@app.post("/predict")
def predict_demand(payload: PredictionRequest):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    
    try:
        data_json = [item.dict() for item in payload.inputs]
        df_input = pd.DataFrame(data_json)
        
        ordered_columns = ['Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'product_id', 'product_name', 'price', 'stock', 'rating']
        df_input = df_input[ordered_columns]

        df_input['product_name'] = df_input['product_name'].astype(np.int32)
        
        predictions = model.predict(df_input)
        
        formatted_predictions = []
        for pred in predictions:
            clean_pred = max(0, round(pred))
            formatted_predictions.append(clean_pred)
            
            # Catat hasil prediksi ke Prometheus
            PREDICTION_DISTRIBUTION.observe(clean_pred)
            
        return {
            "status": "Success",
            "predictions": formatted_predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses inferensi: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)