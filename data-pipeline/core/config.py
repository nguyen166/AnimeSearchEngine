"""
Configuration Management
Load settings from environment variables using Pydantic
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    
    All values can be overridden by setting environment variables.
    """
    
    # ========================================================================
    # AI Service Configuration
    # ========================================================================
    AI_SERVICE_URL: str = "http://localhost:8001/v1/embeddings"
    AI_MODEL: str = "clip"
    AI_SERVICE_TIMEOUT: int = 120
    
    # ========================================================================
    # Milvus Configuration
    # ========================================================================
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_URI: Optional[str] = None  # Alternative to host:port
    MILVUS_COLLECTION: str = "anime_frames"
    VECTOR_DIM: int = 512  # CLIP: 512, SigLIP2: 1152
    MILVUS_USER: Optional[str] = None
    MILVUS_PASSWORD: Optional[str] = None
    
    # ========================================================================
    # Elasticsearch Configuration
    # ========================================================================
    ELASTIC_HOST: str = "localhost"
    ELASTIC_PORT: int = 9200
    ELASTIC_URI: Optional[str] = None  # Alternative to host:port
    ELASTIC_INDEX: str = "anime_frames"
    ELASTIC_USER: Optional[str] = None
    ELASTIC_PASSWORD: Optional[str] = None
    
    # ========================================================================
    # Redis Configuration (Optional - for caching)
    # ========================================================================
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_URI: Optional[str] = None  # redis://localhost:6379/0
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # ========================================================================
    # Processing Configuration
    # ========================================================================
    VIDEO_DIR: str = "./data/raw_videos"
    FRAME_INTERVAL: float = 1.0  # Extract 1 frame per second
    BATCH_SIZE: int = 16  # Batch size for embedding generation
    PROCESSED_LOG: str = "./processed_videos.txt"
    
    # ========================================================================
    # Crawler Configuration
    # ========================================================================
    CRAWLER_OUTPUT_DIR: str = "./data/raw_videos"
    CRAWLER_HEADLESS: bool = True
    CRAWLER_DELAY: int = 5  # Delay between episodes (seconds)
    
    # ========================================================================
    # Logging Configuration
    # ========================================================================
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "data_pipeline.log"
    
    # ========================================================================
    # Pydantic Configuration
    # ========================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # ========================================================================
    # Computed Properties
    # ========================================================================
    
    @property
    def milvus_uri(self) -> str:
        """Get Milvus connection URI"""
        if self.MILVUS_URI:
            return self.MILVUS_URI
        return f"{self.MILVUS_HOST}:{self.MILVUS_PORT}"
    
    @property
    def elastic_uri(self) -> str:
        """Get Elasticsearch connection URI"""
        if self.ELASTIC_URI:
            return self.ELASTIC_URI
        return f"http://{self.ELASTIC_HOST}:{self.ELASTIC_PORT}"
    
    @property
    def redis_uri(self) -> str:
        """Get Redis connection URI"""
        if self.REDIS_URI:
            return self.REDIS_URI
        
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    def log_settings(self):
        """Log current configuration (safe - no passwords)"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 60)
        logger.info("Configuration:")
        logger.info(f"  AI Service: {self.AI_SERVICE_URL}")
        logger.info(f"  Milvus: {self.milvus_uri}")
        logger.info(f"  Elasticsearch: {self.elastic_uri}")
        logger.info(f"  Vector Dim: {self.VECTOR_DIM}")
        logger.info(f"  Batch Size: {self.BATCH_SIZE}")
        logger.info(f"  Frame Interval: {self.FRAME_INTERVAL}s")
        logger.info("=" * 60)


# Singleton instance
settings = Settings()
