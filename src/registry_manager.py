import os
import mlflow
from mlflow.tracking import MlflowClient
import json

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

def manage_model_registry():
    client = MlflowClient()
    experiment_name = "Retail_Demand_Forecasting"
    model_name = "Retail_Demand_XGBoost"

    print("Mencari model terbaik dari eksperimen...")
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print("Eksperimen tidak ditemukan!")
        return
    
    # Cari Model Production Saat Ini (Jika ada)
    current_prod_mae = float('inf')
    try:
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if latest_versions:
            prod_run_id = latest_versions[0].run_id
            prod_run = client.get_run(prod_run_id)
            current_prod_mae = prod_run.data.metrics.get('mae', float('inf'))
            print(f"Model Production saat ini (Versi {latest_versions[0].version}) memiliki MAE: {current_prod_mae:.4f}")
    except Exception as e:
        print("Belum ada model berstatus Production. Model baru akan otomatis dipromosikan.")

    # Cari Run Terakhir/Terbaik dari Pelatihan Baru
    best_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mae ASC"],
        max_results=1
    )
    
    if not best_runs:
        print("Belum ada run/eksperimen yang tersimpan.")
        return

    new_best_run = best_runs[0]
    new_run_id = new_best_run.info.run_id
    new_mae = new_best_run.data.metrics['mae']
    
    print(f"-> Run terbaik ditemukan: {new_run_id} dengan MAE: {new_mae:.4f}")

    # Evaluasi apakah model baru lebih baik dari model Production saat ini
    if new_mae < current_prod_mae:
        print(f"\n✅ PERFORMA MENINGKAT! (Baru: {new_mae:.4f} < Lama: {current_prod_mae:.4f})")
        
        # Registrasi ke Model Registry
        print(f"Mendaftarkan artefak ke MLflow Registry dengan nama '{model_name}'...")
        model_uri = f"runs:/{new_run_id}/model"
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
        export_metadata(new_run_id, version, new_mae)
        
    else:
        print(f"\n❌ PERFORMA MENURUN ATAU SAMA. (Baru: {new_mae:.4f} >= Lama: {current_prod_mae:.4f})")
        print("Model baru ditolak. Model Production lama tetap dipertahankan untuk mencegah model decay.")


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
    print("Metadata diekspor ke models/production_model_meta.json")


if __name__ == "__main__":
    manage_model_registry()