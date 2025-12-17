"""
Embedding Service
HTTP Client để gọi External Embedding Microservice (embedding-service)
"""

import logging
import os
from typing import List, Union, Optional
import base64
import io
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Configuration từ environment variables
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://embedding-service:8000/v1/embeddings")
AI_MODEL = os.getenv("AI_MODEL", "clip")
REQUEST_TIMEOUT = int(os.getenv("AI_SERVICE_TIMEOUT", "30"))  # seconds


class EmbeddingServiceError(Exception):
    """Custom exception cho embedding service errors"""
    pass


class EmbeddingService:
    """
    HTTP Client để gọi external embedding-service microservice.
    
    API Contract:
    - Endpoint: POST /v1/embeddings
    - Request: {"model": "siglip2", "texts": [...], "b64_images": [...], "priority": "normal"}
    - Response: {"object": "list", "data": [{"object": "embedding", "embedding": [...], "index": 0}], "model": "siglip2"}
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._service_url = AI_SERVICE_URL
        self._model = AI_MODEL
        self._timeout = REQUEST_TIMEOUT
        logger.info(f"EmbeddingService initialized - URL: {self._service_url}, Model: {self._model}")
    
    def _image_to_base64(self, image_input: Union[str, bytes, Image.Image]) -> str:
        """
        Chuyển đổi image input thành Base64 string.
        
        Args:
            image_input: Base64 string, bytes, URL, or PIL Image
            
        Returns:
            Base64 encoded string (không có prefix data:image/...)
        """
        if isinstance(image_input, str):
            # Already base64 string
            if image_input.startswith(('http://', 'https://')):
                # Download from URL
                response = requests.get(image_input, timeout=10)
                response.raise_for_status()
                return base64.b64encode(response.content).decode('utf-8')
            else:
                # Remove data URL prefix if present (data:image/jpeg;base64,...)
                if ',' in image_input:
                    return image_input.split(',')[1]
                return image_input
        
        elif isinstance(image_input, bytes):
            return base64.b64encode(image_input).decode('utf-8')
        
        elif isinstance(image_input, Image.Image):
            # Convert PIL Image to base64
            buffer = io.BytesIO()
            # Convert to RGB if needed
            if image_input.mode != 'RGB':
                image_input = image_input.convert('RGB')
            image_input.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def _call_embedding_service(
        self, 
        texts: Optional[List[str]] = None, 
        b64_images: Optional[List[str]] = None,
        priority: str = "normal"
    ) -> List[List[float]]:
        """
        Gọi embedding-service microservice.
        
        Args:
            texts: List of text strings (optional)
            b64_images: List of base64 encoded images (optional)
            priority: Request priority ("normal" or "high")
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingServiceError: Nếu có lỗi kết nối hoặc API trả về error
        """
        payload = {
            "model": self._model,
            "priority": priority
        }
        
        if texts:
            payload["texts"] = texts
        if b64_images:
            payload["b64_images"] = b64_images
        
        try:
            logger.debug(f"Calling embedding service: {self._service_url}")
            
            response = requests.post(
                self._service_url,
                json=payload,
                timeout=self._timeout,
                headers={"Content-Type": "application/json"}
            )
            
            # Handle HTTP errors
            if response.status_code != 200:
                error_msg = f"Embedding service returned {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise EmbeddingServiceError(error_msg)
            
            # Parse response
            result = response.json()
            
            # Extract embeddings từ response
            # Response format: {"object": "list", "data": [{"embedding": [...], "index": 0}, ...], "model": "siglip2"}
            embeddings = []
            for item in result.get("data", []):
                embeddings.append(item.get("embedding", []))
            
            logger.debug(f"Received {len(embeddings)} embeddings from service")
            return embeddings
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Cannot connect to embedding service at {self._service_url}: {e}"
            logger.error(error_msg)
            raise EmbeddingServiceError(error_msg)
        
        except requests.exceptions.Timeout as e:
            error_msg = f"Embedding service timeout after {self._timeout}s: {e}"
            logger.error(error_msg)
            raise EmbeddingServiceError(error_msg)
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Embedding service request failed: {e}"
            logger.error(error_msg)
            raise EmbeddingServiceError(error_msg)
        
        except (KeyError, ValueError) as e:
            error_msg = f"Failed to parse embedding service response: {e}"
            logger.error(error_msg)
            raise EmbeddingServiceError(error_msg)
    
    def encode_image(self, image_input: Union[str, bytes, Image.Image]) -> List[float]:
        """
        Tạo embedding từ hình ảnh thông qua embedding-service.
        
        Args:
            image_input: Image input (base64, bytes, URL, or PIL Image)
            
        Returns:
            Embedding vector (List[float])
        """
        try:
            # Convert image to base64
            b64_image = self._image_to_base64(image_input)
            
            # Call embedding service
            embeddings = self._call_embedding_service(b64_images=[b64_image])
            
            if not embeddings:
                raise EmbeddingServiceError("No embeddings returned from service")
            
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise
    
    def encode_image_base64(self, b64_image: str) -> List[float]:
        """
        Tạo embedding từ base64 encoded image.
        
        Args:
            b64_image: Base64 encoded image string
            
        Returns:
            Embedding vector (List[float])
        """
        try:
            # Remove data URL prefix if present
            if ',' in b64_image:
                b64_image = b64_image.split(',')[1]
            
            # Call embedding service
            embeddings = self._call_embedding_service(b64_images=[b64_image])
            
            if not embeddings:
                raise EmbeddingServiceError("No embeddings returned from service")
            
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"Failed to encode base64 image: {e}")
            raise
    
    def encode_image_bytes(self, image_bytes: bytes) -> List[float]:
        """
        Tạo embedding từ raw image bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Embedding vector (List[float])
        """
        return self.encode_image(image_bytes)

    def encode_text(self, text: str) -> List[float]:
        """
        Tạo embedding từ text thông qua embedding-service.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (List[float])
        """
        try:
            # Call embedding service
            embeddings = self._call_embedding_service(texts=[text])
            
            if not embeddings:
                raise EmbeddingServiceError("No embeddings returned from service")
            
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def encode_batch_images(self, images: List[Union[str, bytes, Image.Image]]) -> List[List[float]]:
        """
        Tạo embeddings cho batch images thông qua embedding-service.
        
        Args:
            images: List of image inputs
            
        Returns:
            List of embedding vectors
        """
        try:
            # Convert all images to base64
            b64_images = [self._image_to_base64(img) for img in images]
            
            # Call embedding service
            embeddings = self._call_embedding_service(b64_images=b64_images)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode batch images: {e}")
            raise
    
    def encode_batch_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embeddings cho batch texts thông qua embedding-service.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            # Call embedding service
            embeddings = self._call_embedding_service(texts=texts)
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode batch texts: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Lấy thông tin về service configuration"""
        return {
            "service_url": self._service_url,
            "model": self._model,
            "timeout": self._timeout,
            "type": "external_microservice"
        }
    
    def health_check(self) -> bool:
        """
        Kiểm tra kết nối đến embedding-service.
        
        Returns:
            True nếu service đang hoạt động, False nếu không
        """
        try:
            # Gửi request test với một text đơn giản
            self.encode_text("test")
            return True
        except Exception as e:
            logger.warning(f"Embedding service health check failed: {e}")
            return False


# Singleton instance
embedding_service = EmbeddingService()
