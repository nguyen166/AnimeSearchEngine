import torch
import open_clip
import logging
from typing import List
from PIL import Image
from configs.config import CONFIG
from .base import BaseEmbeddingService


logger = logging.getLogger(__name__)

class ClipService(BaseEmbeddingService):
    """
    Service class for handling CLIP model inference with enhanced batch processing.
    """
    
    def __init__(self):
        super().__init__(CONFIG["models"]["clip"]["model"])
        self.max_batch_size = CONFIG["models"]["clip"]["max_batch_size"]  # Maximum batch size for processing
        self.cuda_stream = None
        self.device = CONFIG["models"]["clip"]["device"]
        
    async def initialize_model(self):
        """Initialize the CLIP model and preprocessing."""
        await self._initialize_model_base()

    def _load_model(self):
        """Synchronous model loading function with optimizations."""
        logger.info(f"Using device: {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, device=self.device
        )
        self.model.eval()
        
        # PyTorch 2.0+ compilation for better performance
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile for better performance")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        # Initialize CUDA stream for async operations
        if torch.cuda.is_available():
            self.cuda_stream = torch.cuda.Stream()
            logger.info("CUDA stream initialized for async operations")
    
    def _process_batch_in_chunks(self, items, batch_size, process_func):
        """Process items in chunks to avoid memory issues."""
        results = []
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            chunk_results = process_func(chunk)
            results.extend(chunk_results)
        return results
    
    @torch.no_grad()
    def _generate_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with enhanced batching."""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")
        
        if not texts:
            return []
        
        try:
            # Process in chunks if batch is too large
            if len(texts) > self.max_batch_size:
                return self._process_batch_in_chunks(texts, self.max_batch_size, self._generate_text_embeddings)
            
            # Use CUDA stream for async operations if available
            if self.cuda_stream is not None:
                with torch.cuda.stream(self.cuda_stream):
                    text_tokens = open_clip.tokenize(texts).to(self.device, non_blocking=True)
                    text_features = self.model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True).float()
                    result = text_features.cpu().numpy().tolist()
                torch.cuda.synchronize()  # Ensure completion
                return result
            else:
                # Fallback for CPU or when CUDA stream is not available
                text_tokens = open_clip.tokenize(texts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True).float()
                return text_features.cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error generating text embeddings: {e}")
            raise RuntimeError(f"Error generating text embeddings: {e}")
    
    @torch.no_grad()
    def _generate_image_embeddings(self, images: List[Image.Image]) -> List[List[float]]:
        """Generate embeddings for a batch of images with enhanced batching."""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")
        
        if not images:
            return []
        
        try:
            # Process in chunks if batch is too large
            if len(images) > self.max_batch_size:
                return self._process_batch_in_chunks(images, self.max_batch_size, self._generate_image_embeddings)
            
            # Batch all images into a single tensor for efficient processing
            image_tensors = []
            for image in images:
                try:
                    processed_image = self.preprocess(image)
                    image_tensors.append(processed_image)
                except Exception as e:
                    logger.warning(f"Failed to preprocess image: {e}")
                    # Create a zero tensor as fallback
                    image_tensors.append(torch.zeros_like(self.preprocess(Image.new('RGB', (224, 224)))))
            
            if not image_tensors:
                return []
            
            # Stack all images into a single batch tensor
            batch_tensor = torch.stack(image_tensors)
            
            # Use CUDA stream for async operations if available
            if self.cuda_stream is not None:
                with torch.cuda.stream(self.cuda_stream):
                    batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                    embeddings = self.model.encode_image(batch_tensor)
                    embeddings /= embeddings.norm(dim=-1, keepdim=True).float()
                    result = embeddings.cpu().numpy().tolist()
                torch.cuda.synchronize()  # Ensure completion
                return result
            else:
                # Fallback for CPU or when CUDA stream is not available
                batch_tensor = batch_tensor.to(self.device)
                embeddings = self.model.encode_image(batch_tensor)
                embeddings /= embeddings.norm(dim=-1, keepdim=True).float()
                return embeddings.cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error generating image embeddings: {e}")
            raise RuntimeError(f"Error generating image embeddings: {e}")


clip_service = ClipService()