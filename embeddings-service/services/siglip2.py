import logging
from typing import List
from PIL import Image

import torch
from transformers import AutoModel, AutoProcessor

from configs.config import CONFIG
from .base import BaseEmbeddingService


logger = logging.getLogger(__name__)


class Siglip2Service(BaseEmbeddingService):
    """
    Service class for handling SigLIP2 model inference with enhanced batch processing.
    """

    def __init__(self):
        super().__init__(CONFIG["models"]["siglip2"]["model"])
        self.max_batch_size = CONFIG["models"]["siglip2"]["max_batch_size"]
        self.cuda_stream = None
        self.processor = None
        self.image_embedding_dim = None
        self.device = CONFIG["models"]["siglip2"]["device"]

    async def initialize_model(self):
        """Initialize the SigLIP2 model and processor."""
        await self._initialize_model_base()

    def _load_model(self):
        """Synchronous model loading function with optimizations."""
        logger.info(f"Using device: {self.device}")

        # Load model and processor
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)

        # PyTorch compilation for better performance (if available)
        if hasattr(torch, "compile") and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("SigLIP2 model compiled with torch.compile for better performance")
            except Exception as e:
                logger.warning(f"Failed to compile SigLIP2 model: {e}")

        # Initialize CUDA stream for async operations
        if torch.cuda.is_available():
            self.cuda_stream = torch.cuda.Stream()
            logger.info("CUDA stream initialized for async operations (SigLIP2)")

        # Try to infer image embedding dimension upfront with a dummy forward
        try:
            with torch.no_grad():
                dummy = Image.new("RGB", (224, 224))
                inputs = self.processor(images=[dummy], return_tensors="pt")
                if self.device == "cuda":
                    pixel_values = inputs.get("pixel_values", None)
                    if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 4:
                        inputs["pixel_values"] = pixel_values.to(memory_format=torch.channels_last)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self.model.get_image_features(**inputs)
                feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
                if hasattr(feats, "shape") and len(feats.shape) >= 2:
                    self.image_embedding_dim = int(feats.shape[-1])
                    logger.info(f"Detected SigLIP2 image embedding dim: {self.image_embedding_dim}")
        except Exception as e:
            logger.warning(f"Could not infer image embedding dim during load: {e}")

    def _process_batch_in_chunks(self, items, batch_size, process_func):
        """Process items in chunks to avoid memory issues."""
        results = []
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            chunk_results = process_func(chunk)
            results.extend(chunk_results)
        return results

    @torch.no_grad()
    def _generate_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with enhanced batching (SigLIP2)."""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")

        if not texts:
            return []

        try:
            if len(texts) > self.max_batch_size:
                return self._process_batch_in_chunks(texts, self.max_batch_size, self._generate_text_embeddings)

            if self.cuda_stream is not None:
                with torch.cuda.stream(self.cuda_stream):
                    inputs = self.processor(text=texts, return_tensors="pt")
                    inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                    features = self.model.get_text_features(**inputs)
                    features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
                    result = features.cpu().numpy().tolist()
                torch.cuda.synchronize()
                return result
            else:
                inputs = self.processor(text=texts, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                features = self.model.get_text_features(**inputs)
                features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
                return features.cpu().numpy().tolist()

        except Exception as e:
            logger.error(f"Error generating SigLIP2 text embeddings: {e}")
            raise RuntimeError(f"Error generating SigLIP2 text embeddings: {e}")

    @torch.no_grad()
    def _generate_image_embeddings(self, images: List[Image.Image]) -> List[List[float]]:
        """Generate embeddings for a batch of images with enhanced batching (SigLIP2)."""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")

        if not images:
            return []

        try:
            if len(images) > self.max_batch_size:
                return self._process_batch_in_chunks(images, self.max_batch_size, self._generate_image_embeddings)

            # Ensure RGB mode
            rgb_images = [im.convert("RGB") for im in images]

            if self.cuda_stream is not None:
                with torch.cuda.stream(self.cuda_stream):
                    inputs = self.processor(images=rgb_images, return_tensors="pt")
                    if self.device == "cuda":
                        pixel_values = inputs.get("pixel_values", None)
                        if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 4:
                            inputs["pixel_values"] = pixel_values.to(memory_format=torch.channels_last)
                    inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                    feats = self.model.get_image_features(**inputs)
                    feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
                    if self.image_embedding_dim is None and hasattr(feats, "shape") and len(feats.shape) >= 2:
                        self.image_embedding_dim = int(feats.shape[-1])
                    result = feats.cpu().numpy().tolist()
                torch.cuda.synchronize()
                return result
            else:
                inputs = self.processor(images=rgb_images, return_tensors="pt")
                if self.device == "cuda":
                    pixel_values = inputs.get("pixel_values", None)
                    if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 4:
                        inputs["pixel_values"] = pixel_values.to(memory_format=torch.channels_last)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self.model.get_image_features(**inputs)
                feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
                if self.image_embedding_dim is None and hasattr(feats, "shape") and len(feats.shape) >= 2:
                    self.image_embedding_dim = int(feats.shape[-1])
                return feats.cpu().numpy().tolist()

        except Exception as e:
            logger.error(f"Error generating SigLIP2 image embeddings: {e}")
            raise RuntimeError(f"Error generating SigLIP2 image embeddings: {e}")


siglip2_service = Siglip2Service()


