import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Cryptocurrency Fraud Detection"
    DEBUG_MODE: bool = False
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./fraud_detection.db")
    API_V1_STR: str = "/api/v1"
    
    # Model paths
    GNN_MODEL_PATH: str = "models/gnn_model.pt"
    ENSEMBLE_MODEL_PATH: str = "models/ensemble_model.joblib"
    FEATURE_SCALER_PATH: str = "models/feature_scaler.joblib"
    
    # API Keys (replace with your actual keys)
    BLOCKCHAIN_API_KEY: str = os.getenv("BLOCKCHAIN_API_KEY", "your_blockchain_api_key")
    EXCHANGE_API_KEY: str = os.getenv("EXCHANGE_API_KEY", "your_exchange_api_key")
    
    class Config:
        env_file = ".env"

settings = Settings()