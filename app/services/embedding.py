"""
Embedding Service
Quản lý AI model để tạo embeddings từ hình ảnh và text
"""

import logging
from typing import List, Union, Optional
import numpy as np
from PIL import Image
import io
import base64
import requests
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Singleton class để quản lý embedding model"""
    
    _instance = None
    _model = None
    _processor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_model()
    
    def _load_model(self):
        """Load AI model (CLIP hoặc model tương tự)"""
        try:
            # Import torch và transformers khi cần
            import torch
            from transformers import CLIPProcessor, CLIPModel
            
            model_name = settings.MODEL_NAME
            device = settings.DEVICE
            
            logger.info(f"Loading model: {model_name} on {device}")
            
            # Load model và processor
            if settings.MODEL_PATH:
                self._model = CLIPModel.from_pretrained(settings.MODEL_PATH)
                self._processor = CLIPProcessor.from_pretrained(settings.MODEL_PATH)
            else:
                self._model = CLIPModel.from_pretrained(f"openai/{model_name}")
                self._processor = CLIPProcessor.from_pretrained(f"openai/{model_name}")
            
            # Move model to device
            self._model.to(device)
            self._model.eval()
            
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback: Use dummy embeddings for testing
            logger.warning("Using dummy embeddings for testing")
            self._model = None
            self._processor = None
    
    def _preprocess_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """
        Preprocess image input
        
        Args:
            image_input: Base64 string, bytes, URL, or PIL Image
            
        Returns:
            PIL Image
        """
        if isinstance(image_input, Image.Image):
            return image_input
        
        elif isinstance(image_input, str):
            # Check if it's a URL
            if image_input.startswith(('http://', 'https://')):
                response = requests.get(image_input)
                image = Image.open(io.BytesIO(response.content))
            # Assume it's base64
            else:
                # Remove data URL prefix if present
                if ',' in image_input:
                    image_input = image_input.split(',')[1]
                image_bytes = base64.b64decode(image_input)
                image = Image.open(io.BytesIO(image_bytes))
        
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
        
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def encode_image(self, image_input: Union[str, bytes, Image.Image]) -> List[float]:
        """
        Tạo embedding từ hình ảnh
        
        Args:
            image_input: Image input (base64, bytes, URL, or PIL Image)
            
        Returns:
            Embedding vector
        """
        try:
            # Preprocess image
            image = self._preprocess_image(image_input)
            
            # Generate embedding
            if self._model and self._processor:
                import torch
                
                inputs = self._processor(images=image, return_tensors="pt")
                inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = self._model.get_image_features(**inputs)
                    # Normalize
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                embedding = image_features.cpu().numpy().flatten().tolist()
            else:
                # Dummy embedding for testing
                embedding = np.random.rand(settings.VECTOR_DIM).tolist()
                logger.warning("Using dummy embedding")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise
    
    def encode_text(self, text: str) -> List[float]:
        """
        Tạo embedding từ text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            if self._model and self._processor:
                import torch
                
                inputs = self._processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
                    text_features = self._model.get_text_features(**inputs)
                    # Normalize
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                embedding = text_features.cpu().numpy().flatten().tolist()
            else:
                # Dummy embedding for testing
                embedding = np.random.rand(settings.VECTOR_DIM).tolist()
                logger.warning("Using dummy embedding")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def encode_batch_images(self, images: List[Union[str, bytes, Image.Image]]) -> List[List[float]]:
        """
        Tạo embeddings cho batch images
        
        Args:
            images: List of image inputs
            
        Returns:
            List of embedding vectors
        """
        try:
            # Preprocess all images
            processed_images = [self._preprocess_image(img) for img in images]
            
            if self._model and self._processor:
                import torch
                
                inputs = self._processor(images=processed_images, return_tensors="pt")
                inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = self._model.get_image_features(**inputs)
                    # Normalize
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                embeddings = image_features.cpu().numpy().tolist()
            else:
                # Dummy embeddings
                embeddings = [np.random.rand(settings.VECTOR_DIM).tolist() for _ in images]
                logger.warning("Using dummy embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode batch images: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Lấy thông tin về model"""
        return {
            "model_name": settings.MODEL_NAME,
            "device": settings.DEVICE,
            "vector_dim": settings.VECTOR_DIM,
            "loaded": self._model is not None
        }


# Singleton instance
embedding_service = EmbeddingService()
