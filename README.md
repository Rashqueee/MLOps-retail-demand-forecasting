# Retail Demand Forecasting — MLOps Pipeline for Tokopedia Sales Prediction

**Retail Demand Forecasting** is an end-to-end machine learning project aimed at predicting future retail sales/demand using historical data. The system focuses on **time-series regression** using **XGBoost** to estimate product demand in upcoming periods.

This project uses **Tokopedia Product Data** (dynamically ingested) to simulate a real-world e-commerce environment where sales patterns are influenced by intermittent demand, pricing, stock availability, and ratings. 

Moving beyond standard Jupyter Notebook experiments, this project implements a **production-ready MLOps pipeline** featuring automated data ingestion, CI/CD automation, **Data Version Control (DVC)** for remote storage, and **MLflow** for robust experiment tracking and model registry.


## 📋 Project Overview & MLOps Pipeline

The core of this project integrates automated data workflows with independent experiment tracking:

1. **Data Ingestion (`src/ingestion.py`)**
   Simulates chronological streaming data by fetching/scraping snapshots of Tokopedia product data (prices, stock, ratings) and calculating `daily_demand`.
2. **Feature Engineering & Preprocessing (`src/preprocess.py`)**
   Extracts crucial time-based features (e.g., `Year`, `Month`, `DayOfWeek`), processes categorical variables (`product_name`, `product_id`), and handles data anomalies (e.g., clipping negative demand to zero).
3. **Model Training & Experiment Tracking (`src/train.py`)**
   Performs chronologically-aware time-series splitting to train an XGBoost Regressor directly on original values (handling zero-inflated demand). Automatically logs hyperparameters, evaluation metrics (RMSE, MAE), and model artifacts to **MLflow**.
4. **CI/CD Automation (GitHub Actions)**
   Employs a "Code as a Trigger" philosophy. Every push to the repository triggers an automated pipeline that tests the code (pytest), generates mock data, retrains the model, and evaluates it against production thresholds before auto-registering.
5. **Data Versioning (DVC)**
   Tracks datasets and models efficiently, storing the physical files remotely (Google Drive) while keeping the Git repository lightweight.


## 📂 Project Structure

```text
retail-demand-forecasting
│
├─ .github/workflows/       # CI/CD pipeline definitions (mlops-automation.yaml)
├─ .dvc/                    # DVC configuration and cache (ignored in Git)
├─ data/
│  ├─ raw.csv               # Ingested daily snapshots (tracked by DVC)
│  └─ processed.csv         # Cleaned and engineered data ready for training
├─ mlruns/                  # MLflow artifacts (ignored in Git, persisted via Docker volume)
├─ models/                  # Compiled XGBoost models (.json) + production metadata
├─ src/                     # Pipeline source code
│  ├─ ingestion.py          # Data ingestion (Tokopedia)
│  ├─ preprocess.py         # Feature engineering & preprocessing
│  ├─ train.py              # Model training with MLflow tracking
│  ├─ registry_manager.py   # MLflow Model Registry lifecycle management
│  ├─ evaluate_model.py     # Automated evaluation gate for CI/CD
│  └─ main.py               # FastAPI inference API
├─ tests/                   # Unit and integration tests (pytest)
├─ .gitignore
├─ Dockerfile               # Container image for API service
├─ docker-compose.yaml      # Multi-container orchestration manifest
├─ dvc.yaml                 # DVC Pipeline definition
├─ mlflow.db                # MLflow SQLite backend store (ignored in Git)
├─ requirements.txt
└─ README.md

```

## 🛠️ Tech Stack

* **Python 3.10** — Core programming language
* **XGBoost** — High-performance gradient boosting for time-series regression
* **Pandas & NumPy** — Data manipulation and vectorization
* **Scikit-learn** — Preprocessing and evaluation metrics (MAE, RMSE)
* **DVC (Data Version Control)** — Data lineage, versioning, and pipeline orchestration
* **MLflow** — Experiment tracking, metric logging, and model registry
* **FastAPI + Uvicorn** — High-performance REST API for model inference
* **Docker & Docker Compose** — Containerization and multi-service orchestration
* **GitHub Actions** — Continuous Integration & Continuous Deployment (CI/CD)

## 📦 Dataset

This project uses dynamic product tracking data inspired by **Tokopedia**.

Unlike static Kaggle datasets, the data ingestion script dynamically generates or scrapes snapshots of products capturing features like:
`Year`, `Month`, `Day`, `Hour`, `DayOfWeek`, `product_id`, `product_name`, `price`, `stock`, `rating`, and the target variable `daily_demand`.

## 🚀 How to Run the Pipeline (Local Development)

Ensure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

**1. Prepare the Data (DVC Pipeline):**
Run the data ingestion and preprocessing stages using DVC.

```bash
dvc repro
```

**2. Train the Model & Track Experiments (MLflow):**

```bash
python src/train.py
```

**3. Manage Model Registry:**
Evaluate and promote the best model to Production.

```bash
python src/registry_manager.py
```

**4. View MLflow Experiment Dashboard:**

```bash
mlflow ui
# Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser
```

## 🏷️ Model Registry & Inference

This pipeline implements a fully managed lifecycle using the **MLflow Model Registry**. The best-performing model run is programmatically evaluated based on its Mean Absolute Error (MAE), registered, and transitioned across stages (None -> Staging -> Production).

**Currently Active Production Model:**

* **Model Name:** `Tokopedia_Demand_XGBoost`
* **Stage:** `Production`
* **Reason for Selection:** This specific version is selected dynamically due to its superior performance in adapting to zero-inflated intermittent demand data (minimizing MAE) during continuous training rounds.
* **Metadata Tracking:** The exact configuration and lineage of the active production model are stored safely in `models/production_model_meta.json`.

**Inference Mechanism:**
The inference simulation uses `mlflow.pyfunc.load_model` to point dynamically to the `Production` tag via the FastAPI service.

## 🐳 Docker Compose — Multi-Container Orchestration

Seluruh sistem telah diorkestrasi menjadi arsitektur *microservices* menggunakan **Docker Compose** dengan satu perintah.

### Arsitektur Layanan

| Layanan | Deskripsi | Port |
| --- | --- | --- |
| `mlflow-server` | MLflow Tracking Server dengan SQLite backend (`mlflow.db`) | `5000` |
| `api-service` | FastAPI Inference API yang memuat model dari MLflow Registry | `8000` |

### Prasyarat

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) terinstal dan berjalan.
* Port 5000 dan 8000 dalam keadaan kosong (tidak digunakan aplikasi lain).

### Cara Menjalankan Sistem

Menjalankan seluruh ekosistem (MLflow Server + API Inference) di latar belakang:

```bash
docker compose up -d --build

```

*Catatan: `api-service` telah dikonfigurasi menggunakan `depends_on` untuk otomatis menunggu `mlflow-server` menyala.*

### Verifikasi Sistem

```bash
# Cek status kontainer (Pastikan keduanya berstatus "Up")
docker compose ps

# Test API health check
curl http://localhost:8000/
# Expected: {"status":"Online","message":"...","model_loaded":true}

# Test prediksi (Contoh payload Tokopedia)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"Year": 2026, "Month": 6, "Day": 9, "Hour": 14, "DayOfWeek": 1, "product_id": 101, "product_name": "Sepatu Lari Nike", "price": 899000, "stock": 50, "rating": 4.5}]}'
# Expected: {"status":"Success","predictions":[...]}

# Akses MLflow UI
# Buka http://localhost:5000 di browser

```

### Menghentikan Sistem

```bash
docker compose down

```

> **Catatan Persistensi:** Data eksperimen dan *model artifact* tetap aman dan tersimpan di `mlflow.db` (root) dan folder `mlruns/` meskipun kontainer dimatikan, karena telah diamankan menggunakan mekanisme *bind mount volumes* pada Docker.

## 🔄 Future Work

Planned improvements for this project include:

* Implementing **data drift detection** for production monitoring.
* Building a **Streamlit prediction dashboard**.
* ~~Implement End-to-End CI/CD Automation (GitHub Actions).~~ ✅ Done
* ~~Containerizing the application using **Docker**.~~ ✅ Done
* ~~Multi-container orchestration with **Docker Compose**.~~ ✅ Done

## 🤝 Contributing

Contributions are welcome! If you have ideas for improving forecasting accuracy, enhancing feature engineering techniques, experimenting with advanced time-series models (e.g., LSTM, Prophet), or improving the MLOps pipeline architecture, feel free to open an issue or submit a pull request.