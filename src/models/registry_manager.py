import mlflow
from mlflow.tracking import MlflowClient
import json

def manage_model_registry():
    client = MlflowClient()
    experiment_name = "Retail_Demand_Forecasting"
    model_name = "Retail_Demand_XGBoost"

    print("Mencari model terbaik dari eksperimen...")
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print("Eksperimen tidak ditemukan!")
        return

    # Mencari 1 run terbaik berdasarkan MAE terkecil
    best_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mae ASC"],
        max_results=1
    )
    
    if not best_runs:
        print("Belum ada run/eksperimen yang tersimpan.")
        return

    best_run = best_runs[0]
    best_run_id = best_run.info.run_id
    best_mae = best_run.data.metrics['mae']
    
    print(f"-> Run terbaik ditemukan: {best_run_id} dengan MAE: {best_mae:.4f}")

    # Registrasi ke Model Registry (Jika di-run ulang, akan otomatis jadi v2, v3, dst)
    print(f"\nMendaftarkan artefak ke MLflow Registry dengan nama '{model_name}'...")
    model_uri = f"runs:/{best_run_id}/xgboost_model"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    version = model_details.version
    print(f"-> Model terdaftar sebagai Versi: {version}")

    # Transisi ke Staging
    print(f"\nMemindahkan model Versi {version} ke stage 'Staging'...")
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )
    
    # Transisi ke Production (Mengarsipkan versi lama)
    print(f"Memindahkan model Versi {version} ke stage 'Production'...")
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True 
    )
    
    print("\nSiklus hidup model berhasil diperbarui! Model siap digunakan untuk inferensi.")
    
    # Ekspor metadata untuk dilacak oleh Git/DVC
    export_metadata(best_run_id, version, best_mae)


def export_metadata(run_id, version, mae):
    metadata = {
        "model_name": "Retail_Demand_XGBoost",
        "production_run_id": run_id,
        "registry_version": version,
        "mae_score": mae,
        "stage": "Production"
    }
    with open("models/production_model_meta.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("\nMetadata diekspor ke models/production_model_meta.json")

if __name__ == "__main__":
    manage_model_registry()