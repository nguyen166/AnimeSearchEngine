"""
Configuration Management
Quản lý các biến môi trường và cấu hình hệ thống
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Cấu hình ứng dụng từ biến môi trường"""
    
    # FastAPI Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Milvus Configuration
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "anime_frames")
    VECTOR_DIM: int = int(os.getenv("VECTOR_DIM", "512"))
    
    # Elasticsearch Configuration
    ELASTIC_HOST: str = os.getenv("ELASTIC_HOST", "localhost")
    ELASTIC_PORT: int = int(os.getenv("ELASTIC_PORT", "9200"))
    ELASTIC_INDEX: str = os.getenv("ELASTIC_INDEX", "anime_metadata")
    ELASTIC_USER: Optional[str] = os.getenv("ELASTIC_USER")
    ELASTIC_PASSWORD: Optional[str] = os.getenv("ELASTIC_PASSWORD")
    
    # AI Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "clip-vit-base-patch32")
    MODEL_PATH: Optional[str] = os.getenv("MODEL_PATH")
    DEVICE: str = os.getenv("DEVICE", "cpu")  # cpu hoặc cuda
    
    # Search Configuration
    TOP_K: int = int(os.getenv("TOP_K", "10"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    
    # Translation Configuration
    TRANSLATION_MODE: str = os.getenv("TRANSLATION_MODE", "ONLINE")  # Options: "ONLINE", "LOCAL", "GEMINI"
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")  # Required if mode is GEMINI
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # gemini-1.5-flash or gemini-pro
    TRANSLATION_CACHE_TTL: int = int(os.getenv("TRANSLATION_CACHE_TTL", "3600"))  # Cache translations (seconds)
    
    # Data Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    VIDEO_DIR: str = os.getenv("VIDEO_DIR", "./data/videos")
    FRAME_DIR: str = os.getenv("FRAME_DIR", "./data/frames")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton instance
settings = Settings()
