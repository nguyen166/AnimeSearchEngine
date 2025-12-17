import os

class Config:
    """
    Configuration class holding the default model names for different types of requests.
    """
    CLIP_EMBEDDING_MODEL = "hf-hub:apple/DFN5B-CLIP-ViT-H-14"  # Model used for generating embeddings
    MAX_NUM_REQUESTS = int(os.environ.get("MAX_NUM_REQUESTS", "8"))  # Maximum concurrent GPU requests to avoid OOM
