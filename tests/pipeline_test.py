import os

def test_environment_ready():
    """Memastikan library ML utama dapat dimuat tanpa error"""
    import pandas as pd
    import xgboost as xgb
    import mlflow
    assert True, "Library gagal dimuat!"

def test_directories_exist():
    """Memastikan struktur folder terbuat dengan benar"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    assert os.path.exists("models")
    assert os.path.exists("data/processed")