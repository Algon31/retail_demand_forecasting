"""
Global application settings and environment configuration.
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # Environment & Database
    ENV: str = Field(default="development")
    DATABASE_URL: str = Field(default=f"sqlite:///{PROJECT_ROOT / 'retail.db'}")
    
    # API Settings
    API_HOST: str = Field(default="127.0.0.1")
    API_PORT: int = Field(default=8000)
    API_URL: str = Field(default="http://127.0.0.1:8000")
    
    # Data Paths
    DATA_DIR: Path = Field(default=PROJECT_ROOT / "data")
    RAW_DATA_DIR: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    PROCESSED_DATA_DIR: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    FEATURES_DATA_DIR: Path = Field(default=PROJECT_ROOT / "data" / "features")
    MODELS_DIR: Path = Field(default=PROJECT_ROOT / "models")
    
    # Model Artifacts
    MODEL_ARTIFACT_PATH: Path = Field(default=PROJECT_ROOT / "models" / "xgboost_forecaster.joblib")
    BASELINE_ARTIFACT_PATH: Path = Field(default=PROJECT_ROOT / "models" / "baseline_stats.joblib")
    FEATURE_PIPELINE_PATH: Path = Field(default=PROJECT_ROOT / "models" / "feature_pipeline.joblib")
    METRICS_PATH: Path = Field(default=PROJECT_ROOT / "models" / "metrics_summary.json")
    
    # Inventory Defaults
    DEFAULT_SERVICE_LEVEL: float = Field(default=0.95)
    DEFAULT_LEAD_TIME_DAYS: int = Field(default=7)
    DEFAULT_REVIEW_PERIOD_DAYS: int = Field(default=7)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# Ensure directories exist
for directory in [
    settings.DATA_DIR,
    settings.RAW_DATA_DIR,
    settings.PROCESSED_DATA_DIR,
    settings.FEATURES_DATA_DIR,
    settings.MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
