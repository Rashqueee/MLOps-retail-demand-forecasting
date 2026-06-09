import argparse
import pandas as pd
import numpy as np
import os
import sys
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

PROCESSED_DATA_PATH = 'data/processed.csv'
MODEL_DIR = 'models/'

def train_model():
    # Setup argumen untuk hyperparameter tuning
    parser = argparse.ArgumentParser(description="Pelatihan Model XGBoost")
    parser.add_argument("--n_estimators", type=int, default=200, help="Jumlah tree (default: 200)")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Kecepatan belajar (default: 0.1)")
    parser.add_argument("--max_depth", type=int, default=12, help="Kedalaman maksimum tree (default: 12)")
    parser.add_argument("--run_name", type=str, default="Run_Tuning", help="Nama eksperimen di MLflow")
    
    # Membaca argumen yang diberikan dari terminal
    args = parser.parse_args()

    print(f"Memulai proses Model Training dengan MLflow | Run Name: {args.run_name} ...")
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Data {PROCESSED_DATA_PATH} tidak ditemukan.")
        sys.exit(1)
        
    df = pd.read_csv(PROCESSED_DATA_PATH)

    print("Mengonversi tipe data agar kompatibel dengan JSON MLflow Serving...")
    df['product_id'] = df['product_id'].astype(int)
    df['product_name'] = df['product_name'].astype('category').cat.codes 
    
    print("Mempersiapkan data dan melakukan time-series split...")
    df = df.sort_values(by=['Year', 'Month', 'Day', 'Hour'])
    
    fitur_kolom = ['Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'product_id', 'product_name', 'price', 'stock', 'rating']
    
    X = df[fitur_kolom]
    y = df['daily_demand'].clip(lower=0)

    split_index = int(len(df) * 0.90)
    X_train, X_valid = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_valid = y.iloc[:split_index], y.iloc[split_index:]

    mlflow.set_experiment("Retail_Demand_Forecasting")
    
    # Menggunakan nama run dinamis dari argumen
    with mlflow.start_run(run_name=args.run_name):
        
        # Menggunakan parameter dinamis dari argumen
        params = {
            'n_estimators': args.n_estimators,
            'learning_rate': args.learning_rate,
            'max_depth': args.max_depth,
            'random_state': 42,
            'objective': 'reg:squarederror'
        }
        
        # Logging parameter secara otomatis
        for key, value in params.items():
            mlflow.log_param(key, value)
            
        print(f"Sedang melatih model dengan parameter: {params}...")
        xgb_model = xgb.XGBRegressor(**params)
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=False
        )
        
        print("Mengevaluasi model pada data validasi...")
        y_pred = xgb_model.predict(X_valid)

        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        mae = mean_absolute_error(y_valid, y_pred)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        print(f"Validation RMSE: {rmse:.2f}")
        print(f"Validation MAE: {mae:.2f}") 
        
        signature = infer_signature(X_train, y_pred)
        
        print("Menyimpan model ke MLflow...")
        mlflow.xgboost.log_model(xgb_model, artifact_path="model", signature=signature)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        local_model_path = os.path.join(MODEL_DIR, 'xgboost_model.json')
        xgb_model.save_model(local_model_path)
        
        print(f"Training Selesai! Model fisik tersimpan di {local_model_path} untuk DVC.")

if __name__ == "__main__":
    train_model()