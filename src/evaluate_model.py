import mlflow
from mlflow.tracking import MlflowClient
import sys

# Jika MAE (tingkat error) lebih besar dari 10, model dianggap GAGAL.
THRESHOLD_MAE = 10.0 

def evaluate_and_register():
    client = MlflowClient()
    experiment_name = "Retail_Demand_Forecasting"
    model_name = "Tokopedia_Demand_XGBoost"

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print("Eksperimen tidak ditemukan.")
        sys.exit(1)

    # Ambil run terakhir yang baru saja dilatih oleh GitHub Actions
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        print("Tidak ada run ditemukan.")
        sys.exit(1)

    latest_run = runs[0]
    mae_score = latest_run.data.metrics.get('mae', 9999)
    run_id = latest_run.info.run_id

    print(f"Model Run ID: {run_id} | Skor MAE: {mae_score:.2f}")

    # VALIDASI THRESHOLD
    if mae_score > THRESHOLD_MAE:
        print(f"[GAGAL] Performa model memburuk. MAE ({mae_score:.2f}) melebihi ambang batas ({THRESHOLD_MAE}).")
        print("Auto-Registry dihentikan.")
        sys.exit(1) # Gagalkan pipeline GitHub Actions
    else:
        print(f"[SUKSES] MAE ({mae_score:.2f}) memenuhi standar produksi.")
        print("Mendaftarkan ke Model Registry dan transisi ke Staging...")
        
        model_uri = f"runs:/{run_id}/xgboost_model"
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        
        client.transition_model_version_stage(
            name=model_name,
            version=model_details.version,
            stage="Staging"
        )
        print(f"Model Versi {model_details.version} berhasil masuk ke stage Staging!")

if __name__ == "__main__":
    evaluate_and_register()