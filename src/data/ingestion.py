import pandas as pd
import os


SOURCE_DATA_PATH = 'data/external/train.csv'
RAW_DATA_DIR = 'data/raw/'
TRACKER_FILE = 'data/raw/ingestion_tracker.txt'


def get_next_date(source_df, last_date):
    max_date = source_df['Date'].max()

    # Jika tanggal terakhir sudah mencapai atau melewati batas akhir data, kembalikan None
    if last_date >= max_date:
        return None
    
    # Tambah 1 bulan ke depan dari tanggal terakhir
    next_date = last_date + pd.offsets.MonthEnd(1)
    
    # Jika loncatan 1 bulan melewati data terakhir yang tersedia, batasi di max_date
    if next_date > max_date:
        return max_date
        
    return next_date
    

def ingestion():
    print("Memulai proses data ingestion ...")
    
    if not os.path.exists(SOURCE_DATA_PATH):
        print(f"Error: File sumber {SOURCE_DATA_PATH} tidak ditemukan.")
        return

    source_df = pd.read_csv(SOURCE_DATA_PATH, low_memory=False)
    source_df['Date'] = pd.to_datetime(source_df['Date'])
    
    # Tentukan tanggal mulai dataset untuk pengambilan pertama kali
    first_date_in_data = source_df['Date'].min()

    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r') as f:
            last_date_str = f.read().strip()
            last_date = pd.to_datetime(last_date_str)
    else:
        # Jika pertama kali run, anggap belum ada data yang diambil
        last_date = first_date_in_data - pd.Timedelta(days=1)
        
    next_date = get_next_date(source_df, last_date)
    
    if next_date is None:
        print("Semua data sudah habis di-ingest!")
        return
        
    print(f"Menarik data dari {first_date_in_data.strftime('%Y-%m-%d')} hingga {next_date.strftime('%Y-%m-%d')}")
    
    # Ambil semua data dari tanggal paling awal hingga sekarang
    accumulated_data = source_df[
        (source_df['Date'] >= first_date_in_data) & 
        (source_df['Date'] <= next_date)
    ]
    
    # Penamaan file dengan timestamp
    file_timestamp = next_date.strftime('%Y-%m-%d')
    raw_path = os.path.join(RAW_DATA_DIR, f'sales_{file_timestamp}.csv')
    
    # Membuat folder jika belum ada
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Simpan sebagai file baru
    accumulated_data.to_csv(raw_path, index=False)

    # Update tracker
    with open(TRACKER_FILE, 'w') as f:
        f.write(next_date.strftime('%Y-%m-%d'))
        
    print(f"Berhasil menyimpan {len(accumulated_data)} baris transaksi ke {raw_path}")


if __name__ == "__main__":
    ingestion()