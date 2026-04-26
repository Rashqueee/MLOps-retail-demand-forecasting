# Retail Demand Forecasting — MLOps Pipeline for Retail Sales Prediction

**Retail Demand Forecasting** is an end-to-end machine learning project aimed at predicting future retail sales using historical data. The system focuses on **time-series regression** using **XGBoost** to estimate product demand in upcoming periods.

This project uses the **Rossmann Store Sales dataset** to simulate a real-world retail environment where sales patterns are influenced by promotions, holidays, and seasonal trends. 

Moving beyond standard Jupyter Notebook experiments, this project implements a **production-ready MLOps pipeline** featuring automated data ingestion, preprocessing, model training, and **Data Version Control (DVC)** using Google Drive as remote storage.

## 📋 Project Overview & MLOps Pipeline

The core of this project is a fully automated DVC pipeline (`dvc.yaml`) that simulates a real-world continual learning workflow:

1. **Data Ingestion (`src/ingestion.py`)**
   Simulates daily streaming data by fetching chronological snapshots of historical sales data and saving them as versioned accumulated files with timestamps.
2. **Feature Engineering & Preprocessing (`src/preprocess.py`)**
   Merges daily transactions with static store data, extracts crucial time-based features (e.g., `IsWeekend`, `IsPromoMonth`), handles missing values, and drops data-leaking features.
3. **Model Training (`src/train.py`)**
   Performs chronologically-aware time-series splitting to train an XGBoost Regressor. Saves the trained model and logs evaluation metrics (RMSE, MAE).
4. **Data Versioning (DVC)**
   Tracks multi-gigabyte datasets and models efficiently, storing the physical files in Google Drive while keeping the Git repository lightweight.

## 📂 Project Structure

```text
retail-demand-forecasting
│
├─ .dvc/                # DVC configuration and cache (ignored in Git)
├─ config/              # Configuration files
├─ data/
│  ├─ external/         # Raw, immutable source datasets (train.csv, store.csv)
│  ├─ raw/              # Ingested daily snapshots (tracked by DVC)
│  └─ processed/        # Cleaned and engineered data ready for training
├─ metrics/             # Evaluation metrics (e.g., metrics.json)
├─ models/              # Compiled XGBoost models (.json)
├─ notebooks/           # Jupyter notebooks for EDA and prototyping
├─ src/                 # Pipeline source code
│  ├─ ingestion.py
│  ├─ preprocess.py
│  └─ train.py
├─ .gitignore
├─ dvc.yaml             # DVC Pipeline definition
├─ requirements.txt
└─ README.md
```

## 🛠️ Tech Stack

* **Python** — Core programming language
* **XGBoost** — High-performance gradient boosting for time-series regression
* **Pandas & NumPy** — Data manipulation and vectorization
* **Scikit-learn** — Preprocessing and evaluation metrics
* **DVC (Data Version Control)** — Data lineage, versioning, and pipeline orchestration
* **Git** — Source code version control

## 📦 Dataset

This project uses the **Rossmann Store Sales Dataset** from Kaggle.
🔗 [https://www.kaggle.com/c/rossmann-store-sales/data](https://www.kaggle.com/c/rossmann-store-sales/data)

### 📥 How to Download and Prepare Dataset

1. Visit the Kaggle link above and download the dataset.
2. Extract the ZIP file.
3. Place `train.csv` and `store.csv` inside the `data/external/` directory.

   ```text
   retail-demand-forecasting
   ├─ data/
   │  └─ external/
   │     ├─ train.csv
   │     └─ store.csv
   ```

## 🚀 How to Run the Pipeline

Ensure you have installed the required dependencies, including DVC and its Google Drive extension:
```bash
pip install -r requirements.txt
# Ensure dvc and dvc-gdrive are installed
pip install dvc dvc-gdrive
```

**Execute the full pipeline:**
Thanks to DVC, running the entire workflow from data ingestion to model training requires only one command. DVC will intelligently skip stages that haven't changed.
```bash
dvc repro
```

**Sync data to Cloud Storage:**
```bash
dvc push
```

## 🔄 Future Work

Planned improvements for this project include:
* Implementing **Hyperparameter Tuning** via Optuna/GridSearch.
* Adding **MLflow** for robust experiment tracking and model registry.
* Implementing **data drift detection** for production monitoring.
* Building a **Streamlit prediction dashboard**.
* Containerizing the application using **Docker**.

## 🤝 Contributing

Contributions are welcome! If you have ideas for improving forecasting accuracy, enhancing feature engineering techniques, experimenting with advanced time-series models (e.g., LSTM, Prophet), or improving the MLOps pipeline architecture, feel free to open an issue or submit a pull request.