import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.xgboost
import sys

# Path konfigurasi
PROCESSED_DATA_PATH = 'data/processed.csv'
MODEL_DIR = 'models/'

def train_model():
    print("Memulai proses Model Training dengan MLflow ...")
    
    # Load Data Processed
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Data {PROCESSED_DATA_PATH} tidak ditemukan.")
        sys.exit(1)
        
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Konversi kolom Kategorikal untuk XGBoost
    print("Mengonversi tipe data menjadi category...")
    df['product_id'] = df['product_id'].astype('category')
    df['product_name'] = df['product_name'].astype('category')
    
    # Time Series Split (Sorting berdasarkan Waktu ekstraksi)
    print("Mempersiapkan data dan melakukan time-series split...")
    df = df.sort_values(by=['Year', 'Month', 'Day', 'Hour'])
    
    # Target variabel sekarang adalah daily_demand
    X = df.drop(['daily_demand'], axis=1)
    y = df['daily_demand']

    # PENCEGAHAN ERROR: Pastikan tidak ada nilai negatif akibat anomali scraping
    y = y.clip(lower=0)
    
    # Split: 90% data awal untuk Train, 10% data terakhir untuk Validation
    split_index = int(len(df) * 0.90)
    X_train, X_valid = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_valid = y.iloc[:split_index], y.iloc[split_index:]

    # Inisialisasi Eksperimen MLflow
    mlflow.set_experiment("Retail_Demand_Forecasting")
    
    # Mulai pencatatan eksperimen
    with mlflow.start_run(run_name="Run#5"):
        
        # Definisikan Hyperparameter
        params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 10,
            'random_state': 42,
            'objective': 'reg:squarederror',
            'enable_categorical': True
        }
        
        # MLFlow mencatat parameter
        print("Mencatat parameter ke MLflow...")
        for key, value in params.items():
            mlflow.log_param(key, value)
            
        # Melatih Model XGBoost (LANGSUNG menggunakan target asli)
        print("Sedang melatih model...")
        xgb_model = xgb.XGBRegressor(**params)
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=False # Matikan log bawaan XGBoost agar terminal tetap bersih
        )
        
        # Prediksi dan Evaluasi (Hasil langsung berupa angka target asli)
        print("Mengevaluasi model pada data validasi...")
        y_pred = xgb_model.predict(X_valid)

        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        mae = mean_absolute_error(y_valid, y_pred)
        
        # MLFlow mencatat metrik
        print("Mencatat metrik ke MLflow...")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        print(f"Validation RMSE: {rmse:.2f}")
        print(f"Validation MAE: {mae:.2f}") 
        
        # MLFlow mencatat model artefak
        print("Menyimpan model ke MLflow...")
        mlflow.xgboost.log_model(xgb_model, "xgboost_model")
        
        # Menyimpan model secara fisik untuk DVC Pipeline
        os.makedirs(MODEL_DIR, exist_ok=True)
        local_model_path = os.path.join(MODEL_DIR, 'xgboost_model.json')
        xgb_model.save_model(local_model_path)
        
        print(f"Training Selesai! Model fisik tersimpan di {local_model_path} untuk DVC.")

if __name__ == "__main__":
    train_model()