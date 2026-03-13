import pandas as pd
import os

SOURCE_DATA_PATH = '../data/external/train.csv'
RAW_DATA_PATH = '../data/raw/daily_sales.csv'
TRACKER_FILE = '../data/raw/ingestion_tracker.txt'

def get_next_date(source_df, last_date):
    future_dates = source_df[source_df['Date'] > last_date]['Date'].unique()
    if len(future_dates) == 0:
        return None
    return future_dates.min()

def ingestion():
    print("Memulai proses data ingestion ...")
    
    if not os.path.exists(SOURCE_DATA_PATH):
        print(f"Error: File sumber {SOURCE_DATA_PATH} tidak ditemukan.")
        return

    source_df = pd.read_csv(SOURCE_DATA_PATH, low_memory=False)
    source_df['Date'] = pd.to_datetime(source_df['Date'])
    
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r') as f:
            last_date_str = f.read().strip()
            last_date = pd.to_datetime(last_date_str)
    else:
        last_date = source_df['Date'].min() - pd.Timedelta(days=1)
        
    next_date = get_next_date(source_df, last_date)
    
    if next_date is None:
        print("Semua data sudah habis di-ingest!")
        return
        
    print(f"Menarik data untuk tanggal: {next_date.strftime('%Y-%m-%d')}")
    
    daily_data = source_df[source_df['Date'] == next_date]
    
    if os.path.exists(RAW_DATA_PATH):
        daily_data.to_csv(RAW_DATA_PATH, mode='a', header=False, index=False)
    else:
        daily_data.to_csv(RAW_DATA_PATH, mode='w', header=True, index=False)
        
    with open(TRACKER_FILE, 'w') as f:
        f.write(next_date.strftime('%Y-%m-%d'))
        
    print(f"Berhasil menyimpan {len(daily_data)} baris transaksi ke {RAW_DATA_PATH}")

if __name__ == "__main__":
    ingestion()