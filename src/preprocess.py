import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import glob


STORE_DATA_PATH = '../data/external/store.csv'
PROCESSED_DATA_PATH = '../data/processed/processed_sales.csv'


def preprocess_data():
    print("Memulai proses data preprocessing ...")
    
    list_of_files = glob.glob('../data/raw/sales_*.csv')
    latest_file = max(list_of_files, key=os.path.getctime)
        
    print("Memuat data terbaru ({latest_file})...")
    df_train = pd.read_csv(latest_file, low_memory=False)
    df_store = pd.read_csv(STORE_DATA_PATH, low_memory=False)
    
    # Hapus kolom Customers untuk mencegah data leakage
    if 'Customers' in df_train.columns:
        print("Mengahapus kolom 'Customers' untuk mencegah data leakage...")
        df_train = df_train.drop('Customers', axis=1)
    
    print("Menggabungkan data transaksi dengan data toko...")
    df = pd.merge(df_train, df_store, how='left', on='Store')
    
    # Feature Engineering: Tanggal
    print("Mengekstrak fitur tanggal...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    
    # Handling Missing Values
    print("Menangani missing values...")
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
    
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
    df['PromoInterval'] = df['PromoInterval'].fillna('None')
    
    # FEATURE ENGINEERING: PromoInterval
    # Cek apakah bulan transaksi ini termasuk bulan promo bagi toko tersebut
    print("Memproses PromoInterval...")
    month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
                 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    df['MonthStr'] = df['Month'].map(month_map)
    
    # Buat kolom baru: 1 jika sedang dalam bulan Promo2, 0 jika tidak
    df['IsPromoMonth'] = df.apply(
        lambda x: 1 if isinstance(x['PromoInterval'], str) and x['MonthStr'] in x['PromoInterval'] else 0, 
        axis=1
    )
    # Setelah diekstrak logikanya, PromoInterval asli bisa di-drop
    df = df.drop(['PromoInterval', 'MonthStr'], axis=1)

    # Fitur apakah hari ini weekend (Sabtu=6, Minggu=7)
    print("Memproses Weekend...")
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 6 else 0)

    # Encoding Categorical Variables
    print("Melakukan encoding pada kolom kategorikal...")
    label_encoder = LabelEncoder()
    categorical_cols = ['StateHoliday', 'StoreType', 'Assortment'] # PromoInterval sudah dihapus
    
    for col in categorical_cols:
        df[col] = df[col].astype(str) # Memastikan '0' (angka) dan 'a' (huruf) menjadi tipe string seragam
        df[col] = label_encoder.fit_transform(df[col])
        
    # Memfilter toko yang tutup atau memiliki 0 sales (Sesuai metrik evaluasi kompetisi)
    print("Memfilter toko yang tutup atau memiliki 0 sales...")
    if 'Sales' in df.columns:
        df = df[(df['Open'] != 0) & (df['Sales'] > 0)]
    
    # Drop kolom Date karena fiturnya sudah dipecah
    df = df.drop(['Date'], axis=1)
    
    # Menyimpan Data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Berhasil menyimpan {len(df)} baris data bersih ke {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    preprocess_data()