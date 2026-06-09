# Menggunakan base image python yang ringan
FROM python:3.10-slim

# Menetapkan environment variables agar output log muncul real-time di terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Menetapkan direktori kerja di dalam kontainer
WORKDIR /app

# Menginstal dependensi sistem yang mungkin dibutuhkan oleh library ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Menyalin file requirements.txt terlebih dahulu (agar proses build lebih cepat dengan caching)
COPY requirements.txt .

# Menginstal dependensi Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh isi proyek ke dalam direktori kerja
COPY . .

# Mengekspos port 8000 untuk API FastAPI
EXPOSE 8000

# Perintah untuk menjalankan FastAPI dengan Uvicorn
# Kita menunjuk ke src.main karena file main.py ada di dalam folder src
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]