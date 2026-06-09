import pandas as pd
import os

RAW_DATA_PATH = 'data/raw.csv'
PROCESSED_DATA_PATH = 'data/processed.csv'

def preprocess_tokopedia():
    print("Memulai preprocessing data Tokopedia...")
    if not os.path.exists(RAW_DATA_PATH):
        print("Data raw tidak ditemukan.")
        return

    df = pd.read_csv(RAW_DATA_PATH)

    # 1. Konversi waktu
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 2. Urutkan berdasarkan produk lalu waktu (SANGAT PENTING untuk hitung selisih)
    df = df.sort_values(by=['product_id', 'timestamp'])

    # 3. Hitung Penjualan Aktual (Delta/Selisih dari akumulasi 'sold')
    # Mengelompokkan per produk, lalu menghitung selisih baris saat ini dengan sebelumnya
    df['daily_demand'] = df.groupby('product_id')['sold'].diff().fillna(0)
    
    # Karena kita mungkin scraping beberapa kali sehari, kita buang data yang tidak ada pergerakan
    # atau Anda bisa melakukan resample menjadi data per hari/per jam
    
    # 4. Ekstrak Fitur Waktu
    df['Year'] = df['timestamp'].dt.year
    df['Month'] = df['timestamp'].dt.month
    df['Day'] = df['timestamp'].dt.day
    df['Hour'] = df['timestamp'].dt.hour
    df['DayOfWeek'] = df['timestamp'].dt.dayofweek

    # 5. Konversi Tipe Data Kategorikal
    df['product_id'] = df['product_id'].astype('category')
    df['product_name'] = df['product_name'].astype('category')

    # 6. Buang kolom yang tidak relevan untuk ML
    # Kita buang 'sold' (akumulasi) dan 'timestamp' (sudah dipecah)
    df_processed = df.drop(['timestamp', 'sold'], axis=1)

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Preprocessing selesai. Data disimpan di:", PROCESSED_DATA_PATH)

if __name__ == "__main__":
    preprocess_tokopedia()