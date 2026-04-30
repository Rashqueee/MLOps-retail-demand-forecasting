import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import mlflow
import mlflow.xgboost

# Path konfigurasi
PROCESSED_DATA_PATH = 'data/processed/processed_sales.csv'
MODEL_DIR = 'models/'

def train_model():
    print("Memulai proses Model Training dengan MLflow ...")
    
    # Load Data Processed
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Data {PROCESSED_DATA_PATH} tidak ditemukan.")
        return
        
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Konversi kembali kolom Store menjadi tipe data 'category'
    print("Mengonversi tipe data Store menjadi category...")
    df['Store'] = df['Store'].astype('category')
    
    # Time Series Split (Sorting kronologis)
    print("Mempersiapkan data dan melakukan time-series split...")
    df = df.sort_values(by=['Year', 'Month', 'Day'])
    
    X = df.drop(['Sales'], axis=1)
    y = df['Sales']
    
    # Split: 90% data awal untuk Train, 10% data terakhir untuk Validation
    split_index = int(len(df) * 0.90)
    X_train, X_valid = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_valid = y.iloc[:split_index], y.iloc[split_index:]
    
    # Log tranformation pada target 
    print("Menerapkan transformasi logaritmik pada variabel target...")
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    # Inisialisasi Eksperimen MLflow (akan membuat folder mlruns/ jika dijalankan secara lokal)
    mlflow.set_experiment("Retail_Demand_Forecasting")
    
    # Mulai pencatatan eksperimen
    with mlflow.start_run(run_name="Run#1"):
        
        # Definisikan Hyperparameter
        params = {
            'n_estimators': 50,
            'learning_rate': 0.05,
            'max_depth': 4,
            'random_state': 42,
            'objective': 'reg:squarederror',
            'enable_categorical': True
        }
        
        # MLFlow mencatat parameter
        print("Mencatat parameter ke MLflow...")
        for key, value in params.items():
            mlflow.log_param(key, value)
            
        # Melatih Model XGBoost
        print("Sedang melatih model...")
        xgb_model = xgb.XGBRegressor(**params)
        
        # Training menggunakan target yang sudah di-log
        xgb_model.fit(
            X_train, y_train_log,
            eval_set=[(X_train, y_train_log), (X_valid, y_valid_log)],
            verbose=False # Matikan log bawaan XGBoost agar terminal tetap bersih
        )
        
        # Prediksi dan Evaluasi
        print("Mengevaluasi model pada data validasi...")
        y_pred_log = xgb_model.predict(X_valid) # Hasil prediksi masih dalam bentuk logaritmik
        
        # Kembalikan ke nilai asli (inverse transform) sebelum menghitung error
        y_pred = np.expm1(y_pred_log)

        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        mape = mean_absolute_percentage_error(y_valid, y_pred)
        
        # MLFlow mencatat metrik
        print("Mencatat metrik ke MLflow...")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mape", mape)
        
        print(f"Validation RMSE: {rmse:.2f}")
        print(f"Validation MAPE: {mape:.2%}") 
        
        # MLFlow mencatat model artefak
        print("Menyimpan model ke MLflow...")
        mlflow.xgboost.log_model(xgb_model, "xgboost_model")
        
        # Menyimpan model secara fisik untuk DVC Pipeline
        os.makedirs(MODEL_DIR, exist_ok=True)
        local_model_path = os.path.join(MODEL_DIR, 'xgboost_sales_model.json')
        xgb_model.save_model(local_model_path)
        
        print(f"Training Selesai! Model fisik tersimpan di {local_model_path} untuk DVC.")

if __name__ == "__main__":
    train_model()