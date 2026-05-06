import mlflow
from mlflow.tracking import MlflowClient


def manage_model_registry():
    client = MlflowClient()
    experiment_name = "Retail_Demand_Forecasting"
    model_name = "Rossmann_Sales_XGBoost"

    print("Mencari model terbaik dari eksperimen...")
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print("Eksperimen tidak ditemukan!")
        return

    # Mencari 1 run terbaik berdasarkan MAPE terkecil
    best_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mape ASC"],
        max_results=1
    )
    
    best_run = best_runs[0]
    best_run_id = best_run.info.run_id
    best_mape = best_run.data.metrics['mape']
    
    print(f"-> Run terbaik ditemukan: {best_run_id} dengan MAPE: {best_mape:.2%}")

    # Jika nama model belum ada, akan dibuat sebagai v1. 
    # Jika sudah ada (dari percobaan sebelumnya), otomatis menjadi versi berikutnya (v2, v3, dst.)
    print(f"\nMendaftarkan artefak ke MLflow Registry dengan nama '{model_name}'...")
    model_uri = f"runs:/{best_run_id}/xgboost_model"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    version = model_details.version
    print(f"-> Model terdaftar sebagai Versi: {version}")


    # Naikkan ke Staging dulu (untuk testing), lalu ke Production
    print(f"\nMemindahkan model Versi {version} ke stage 'Staging'...")
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )
    
    print(f"Memindahkan model Versi {version} ke stage 'Production'...")
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True # Otomatis mengarsipkan/menurunkan versi Production lama
    )
    
    print("\nSiklus hidup model berhasil diperbarui! Model siap digunakan untuk inferensi.")
    
    # Ekspor metadata untuk DVC
    export_metadata_for_dvc(best_run_id, version, best_mape)


def export_metadata_for_dvc(run_id, version, mape):
    import json
    metadata = {
        "production_run_id": run_id,
        "registry_version": version,
        "mape_score": mape
    }
    with open("data/processed/production_model_meta.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("\nMetadata diekspor ke production_model_meta.json untuk dilacak oleh DVC.")


if __name__ == "__main__":
    manage_model_registry()