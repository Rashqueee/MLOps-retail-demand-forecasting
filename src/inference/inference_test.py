import mlflow.pyfunc
import pandas as pd
import numpy as np


def test_production_inference():
    model_name = "Rossmann_Sales_XGBoost"
    stage = "Production"
    
    print(f"Menarik model '{model_name}' dari stage '{stage}'...")
    
    # MLflow akan secara otomatis mencari model mana pun yang sedang berstatus Production
    model_uri = f"models:/{model_name}/{stage}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    print("Model berhasil dimuat ke memory!")
    print("Tipe model:", type(loaded_model))
    
    # --- SIMULASI INFERENSI ---
    # Kita ambil 5 baris pertama dari data processed sebagai contoh payload (request) dari user
    print("\nMensimulasikan inferensi pada data baru...")
    df_dummy = pd.read_csv('data/processed/processed_sales.csv').head(5)
    
    # Pisahkan target jika ada
    if 'Sales' in df_dummy.columns:
        df_dummy = df_dummy.drop('Sales', axis=1)
        
    # Pastikan tipe data sama dengan saat training (Handling category)
    if 'Store' in df_dummy.columns:
        df_dummy['Store'] = df_dummy['Store'].astype('category')
        
    # Lakukan prediksi (Hasil masih berupa log)
    log_predictions = loaded_model.predict(df_dummy)
    
    # Kembalikan dengan inverse transform
    actual_sales_predictions = np.expm1(log_predictions)
    
    print("\nHasil Prediksi Sales untuk 5 transaksi:")
    for i, pred in enumerate(actual_sales_predictions):
        print(f"Transaksi {i+1}: Prediksi Penjualan = {pred:.2f}")

if __name__ == "__main__":
    test_production_inference()