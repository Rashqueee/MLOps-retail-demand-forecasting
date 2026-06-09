# Retail Demand Forecasting — MLOps Pipeline for Tokopedia Sales Prediction

**Retail Demand Forecasting** is an end-to-end machine learning project aimed at predicting future retail sales/demand using historical data. The system focuses on **time-series regression** using **XGBoost** to estimate product demand in upcoming periods.

This project uses **Tokopedia Product Data** (dynamically ingested) to simulate a real-world e-commerce environment where sales patterns are influenced by intermittent demand, pricing, stock availability, and ratings. 

Moving beyond standard Jupyter Notebook experiments, this project implements a **production-ready MLOps pipeline** featuring automated data ingestion, CI/CD automation, **Data Version Control (DVC)** for remote storage, **MLflow** for robust experiment tracking, and a full **Observability Stack** (Prometheus & Grafana) to monitor model performance in real-time.


## 📋 Project Overview & MLOps Pipeline

The core of this project integrates automated data workflows with independent experiment tracking and proactive monitoring:

1. **Data Ingestion (`src/ingestion.py`)**
   Simulates chronological streaming data by fetching/scraping snapshots of Tokopedia product data (prices, stock, ratings) and calculating `daily_demand`.
2. **Feature Engineering & Preprocessing (`src/preprocess.py`)**
   Extracts crucial time-based features (e.g., `Year`, `Month`, `DayOfWeek`), applies Label Encoding to categorical variables (e.g., `product_name`), and handles data anomalies.
3. **Model Training & Experiment Tracking (`src/train.py`)**
   Performs chronologically-aware time-series splitting to train an XGBoost Regressor. Automatically logs hyperparameters, evaluation metrics (RMSE, MAE), model signatures, and artifacts to **MLflow**. Features dynamic CLI arguments for rapid tuning.
4. **CI/CD Automation (GitHub Actions)**
   Employs a "Code as a Trigger" philosophy. Every push to the repository triggers an automated pipeline that tests the code (pytest), generates mock data, retrains the model, and evaluates it.
5. **Data Versioning (DVC)**
   Tracks datasets and models efficiently, storing the physical files remotely (Google Drive) while keeping the Git repository lightweight.
6. **Deployment & Scaling**
   Wraps the production model in a high-performance **FastAPI** service, containerized via **Docker**, and horizontally scaled using Docker Compose replicas to handle high workloads.
7. **Observability (Prometheus & Grafana)**
   Actively monitors API throughput, inference latency, hardware utilization, and potential **Data Drift** by analyzing the distribution of the model's predictions in real-time.


## 📂 Project Structure

```text
retail-demand-forecasting
│
├─ .github/workflows/       # CI/CD pipeline definitions
├─ .dvc/                    # DVC configuration and cache (ignored in Git)
├─ data/
│  ├─ raw.csv               # Ingested daily snapshots (tracked by DVC)
│  └─ processed.csv         # Cleaned and engineered data ready for training
├─ mlruns/                  # MLflow artifacts (ignored in Git, persisted via volume)
├─ models/                  # Compiled XGBoost models (.json) + production metadata
├─ src/                     # Pipeline source code
│  ├─ ingestion.py          # Data ingestion (Tokopedia)
│  ├─ preprocess.py         # Feature engineering & preprocessing
│  ├─ train.py              # Model training with MLflow tracking & argparse
│  ├─ registry_manager.py   # MLflow Model Registry lifecycle management
│  ├─ evaluate_model.py     # Automated evaluation gate for CI/CD
│  └─ main.py               # FastAPI inference API with Prometheus Instrumentator
├─ tests/                   # Unit and integration tests (pytest)
├─ .gitignore
├─ Dockerfile               # Container image for API service
├─ docker-compose.yaml      # Multi-container orchestration manifest
├─ dvc.yaml                 # DVC Pipeline definition
├─ mlflow.db                # MLflow SQLite backend store (ignored in Git)
├─ prometheus.yml           # Prometheus scraping configuration
├─ requirements.txt
└─ README.md
```


## 🛠️ Tech Stack

* **Python 3.10** — Core programming language
* **XGBoost & Scikit-learn** — High-performance modeling and evaluation
* **Pandas & NumPy** — Data manipulation and vectorization
* **DVC (Data Version Control)** — Data lineage and pipeline orchestration
* **MLflow** — Experiment tracking, metric logging, and Model Registry
* **FastAPI + Uvicorn** — REST API for model inference
* **Docker & Docker Compose** — Containerization and multi-service orchestration
* **Prometheus & Grafana** — Metric scraping and Observability dashboards
* **GitHub Actions** — Continuous Integration & Continuous Deployment (CI/CD)


## 🐳 Docker Compose — Multi-Container Orchestration

The entire ecosystem is orchestrated into a microservices architecture using **Docker Compose** with a single command.

### Service Architecture

| Service | Description | Port |
| --- | --- | --- |
| `mlflow-server` | MLflow Tracking Server with SQLite backend (`mlflow.db`) | `5000` |
| `api-service` | FastAPI Inference API (Horizontally scaled to 3 instances) | `8000-8002` |
| `prometheus` | Time-series database scraping metrics from the API | `9090` |
| `grafana` | Visual dashboard for system and model observability | `3000` |

### Running the System

To launch the tracking server, load balancer, API replicas, and monitoring stack in the background:

```bash
docker compose up -d --build
```


## 🧪 Hyperparameter Tuning & Experimentation

The training script (`src/train.py`) is fully equipped with `argparse`, allowing seamless hyperparameter tuning directly from the command line **without needing to rebuild the Docker container**.

You can execute multiple experimental runs inside the running container to find the optimal model configuration. All parameters and metrics will be automatically logged to the MLflow UI (`http://localhost:5000`).

**Examples of Rapid Experimentation:**

```bash
# Run 1: Default Parameters
docker compose exec api-service python src/train.py --run_name "Run#1_Baseline"

# Run 2: Fast & Shallow Model (Prevent Overfitting)
docker compose exec api-service python src/train.py --run_name "Run#2_Shallow" --n_estimators 100 --learning_rate 0.1 --max_depth 6

# Run 3: Slow & Deep (High Precision Tuning)
docker compose exec api-service python src/train.py --run_name "Run#3_Deep" --n_estimators 500 --learning_rate 0.01 --max_depth 10
```

Once you find the best model via the MLflow UI, you can promote it to production by running:

```bash
docker compose exec api-service python src/registry_manager.py
docker compose restart api-service
```


## ⚖️ Horizontal Scaling & Inference Testing

To anticipate high traffic, the system implements **Horizontal Scaling** dynamically utilizing Docker Compose's `deploy: replicas` feature. The load is automatically balanced across available ports.

**Scaling On-the-Fly (Dynamic Scaling):**

```bash
docker compose up -d --scale api-service=5
```

**Testing the Inference API:**
The API validates inputs strictly according to the MLflow Model Signature. Below is an example payload representing a high-traffic "Flash Sale" scenario:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "Year": 2026,
        "Month": 6,
        "Day": 25,
        "Hour": 20,
        "DayOfWeek": 3,
        "product_id": 101,
        "product_name": 1,
        "price": 49000,
        "stock": 1000,
        "rating": 4.9
      }
    ]
  }'
```


## 📊 Observability & Proactive Monitoring

This pipeline is fully instrumented using `prometheus-fastapi-instrumentator` to expose operational and business metrics, allowing proactive detection of **System Decay** and **Data Drift**.

### Accessing the Dashboards:

1. **Prometheus Targets:** Visit `http://localhost:9090/targets` to verify the scraper is actively pulling data from all API replicas.
2. **Grafana Dashboard:** Visit `http://localhost:3000` (Credentials: `admin` / `admin`).

### Key Metrics Tracked:

* **Throughput (Requests Per Second):** Monitors the load distributed across the replicas.
* **Average Inference Latency:** Tracks API response times to detect system bottlenecks.
* **RAM Utilization:** Monitors memory leaks within the containerized Python processes.
* **Model Prediction Output (Custom Histogram):** Tracks the real-time distribution of the predicted demand values. A sudden shift in this metric serves as an early warning indicator for **Data Drift**, signaling that the model may require retraining.


## 🤝 Contributing

Contributions are welcome! If you have ideas for improving forecasting accuracy, enhancing feature engineering techniques, experimenting with advanced time-series models (e.g., LSTM, Prophet), or improving the MLOps pipeline architecture, feel free to open an issue or submit a pull request.