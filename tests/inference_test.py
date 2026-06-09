import mlflow.pyfunc
import pandas as pd

def test_production_inference():
    model_name = "Retail_Demand_XGBoost"
    stage = "Production"
    
    print(f"Menarik model '{model_name}' dari stage '{stage}'...")
    
    # MLflow secara otomatis mencari model yang berstatus Production
    model_uri = f"models:/{model_name}/{stage}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    print("Model berhasil dimuat ke memory!")
    
    # --- SIMULASI INFERENSI ---
    print("\nMensimulasikan inferensi pada data baru...")
    df_dummy = pd.read_csv('data/processed.csv').head(5)
    
    # Buang target 'daily_demand' karena kita akan memprediksinya
    if 'daily_demand' in df_dummy.columns:
        df_dummy = df_dummy.drop('daily_demand', axis=1)
        
    # Pastikan tipe data kategorikal sama dengan saat training
    df_dummy['product_id'] = df_dummy['product_id'].astype('category')
    df_dummy['product_name'] = df_dummy['product_name'].astype('category')
        
    # Lakukan prediksi (Tanpa expm1 karena tidak pakai log transform)
    predictions = loaded_model.predict(df_dummy)
    
    print("\nHasil Prediksi Demand Harian (Penjualan) untuk 5 baris pertama:")
    for i, pred in enumerate(predictions):
        # Memastikan output tidak negatif dan dibulatkan ke bilangan bulat
        pred_clean = max(0, round(pred))
        produk = df_dummy['product_name'].iloc[i]
        print(f"Transaksi {i+1} | {produk[:25]}... : Prediksi = {pred_clean} unit")

if __name__ == "__main__":
    test_production_inference()