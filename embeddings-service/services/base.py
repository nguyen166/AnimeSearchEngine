import os
import time
import logging
import base64
import io
import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from queue import Queue
from threading import Thread, Lock, Event, Semaphore, current_thread
from PIL import Image

logger = logging.getLogger(__name__)


class BaseEmbeddingService(ABC):
    """
    Abstract base class for embedding services with common queue management and task processing.
    
    This class provides:
    - Queue-based task processing with worker threads
    - Async model initialization
    - Task status tracking and result storage
    - Base64 image processing utilities
    - Graceful shutdown handling
    """
    
    def __init__(self, model_name: str, max_concurrent_requests: Optional[int] = None):
        """
        Initialize the base embedding service.
        
        Args:
            model_name: Name of the model to use for embeddings
            max_concurrent_requests: Maximum number of concurrent GPU requests (defaults to Config.MAX_NUM_REQUESTS)
        """
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.device = None
        self.is_initialized = False
        self.initialization_lock = Lock()
        
        # Concurrency control
        self.max_concurrent_requests = max_concurrent_requests or 1
        self.gpu_semaphore = Semaphore(self.max_concurrent_requests)
        
        # Queue management
        self.task_queue = Queue()
        self.result_store: Dict[str, Dict[str, Any]] = {}
        self.result_lock = Lock()
        self.worker_threads: List[Thread] = []
        self.is_processing = False
        self.shutdown_event = Event()
    
    @abstractmethod
    async def initialize_model(self):
        """
        Initialize the specific model and preprocessing.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def _load_model(self):
        """
        Synchronous model loading function.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def _generate_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        Must be implemented by subclasses.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def _generate_image_embeddings(self, images: List[Image.Image]) -> List[List[float]]:
        """
        Generate embeddings for a batch of images.
        Must be implemented by subclasses.
        
        Args:
            images: List of PIL Images to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    async def _initialize_model_base(self):
        """
        Base initialization logic that can be called by subclasses.
        Handles the common initialization pattern.
        """
        if self.is_initialized:
            return
            
        with self.initialization_lock:
            if self.is_initialized:
                return
                
            try:
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model)
                self.is_initialized = True
                logger.info(f"{self.__class__.__name__} model initialized successfully on device: {self.device}")
                
                # Start worker thread for queue processing
                self.start_worker()
                
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                raise e
    
    def start_worker(self):
        """Start multiple worker threads for processing queued tasks."""
        if not self.is_processing:
            self.is_processing = True
            # Start multiple worker threads up to max_concurrent_requests
            for i in range(self.max_concurrent_requests):
                worker_thread = Thread(target=self._process_queue, daemon=True, name=f"worker-{i}")
                worker_thread.start()
                self.worker_threads.append(worker_thread)
            logger.info(f"Started {self.max_concurrent_requests} worker threads for {self.__class__.__name__} queue processing")
    
    def _process_queue(self):
        """Worker thread function to process tasks from the queue."""
        thread_name = current_thread().name
        logger.info(f"Worker thread {thread_name} started")
        
        while not self.shutdown_event.is_set():
            try:
                if not self.task_queue.empty():
                    task = self.task_queue.get()
                    task_id = task['task_id']
                    texts = task.get('texts', [])
                    images = task.get('images', [])
                    
                    # Acquire GPU semaphore before processing
                    logger.debug(f"Thread {thread_name}: Acquiring GPU semaphore for task {task_id}")
                    with self.gpu_semaphore:
                        try:
                            # Update status to processing
                            with self.result_lock:
                                if task_id in self.result_store:
                                    self.result_store[task_id]['status'] = 'processing'
                            
                            logger.info(f"Thread {thread_name}: Processing task {task_id} with {len(texts)} texts and {len(images)} images")
                            
                            # Process texts and images
                            embeddings = []
                            
                            # Process text embeddings
                            if texts:
                                text_embeddings = self._generate_text_embeddings(texts)
                                embeddings.extend(text_embeddings)
                            
                            # Process image embeddings
                            if images:
                                image_embeddings = self._generate_image_embeddings(images)
                                embeddings.extend(image_embeddings)
                            
                            # Store results
                            with self.result_lock:
                                self.result_store[task_id] = {
                                    'status': 'completed',
                                    'embeddings': embeddings,
                                    'error': None,
                                    'completed_at': time.time()
                                }
                            
                            logger.info(f"Thread {thread_name}: Task {task_id} completed successfully")
                            
                        except Exception as e:
                            logger.error(f"Thread {thread_name}: Error processing task {task_id}: {e}")
                            with self.result_lock:
                                self.result_store[task_id] = {
                                    'status': 'failed',
                                    'embeddings': None,
                                    'error': str(e),
                                    'completed_at': time.time()
                                }
                    
                    self.task_queue.task_done()
                else:
                    # Sleep briefly if queue is empty
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Thread {thread_name}: Error in queue processing: {e}")
                time.sleep(1)
        
        logger.info(f"Worker thread {thread_name} shutting down")
    
    async def add_task(self, task_id: str, texts: List[str] = None, images: List[Image.Image] = None) -> str:
        """
        Add a new embedding task to the queue.
        
        Args:
            task_id: Unique identifier for the task
            texts: Optional list of text strings to embed
            images: Optional list of PIL Images to embed
            
        Returns:
            The task ID
        """
        if not self.is_initialized:
            await self.initialize_model()
        
        texts = texts or []
        images = images or []
        
        # Initialize result entry
        with self.result_lock:
            self.result_store[task_id] = {
                'status': 'queued',
                'embeddings': None,
                'error': None,
                'created_at': time.time()
            }
        
        # Add task to queue
        task = {
            'task_id': task_id,
            'texts': texts,
            'images': images
        }
        self.task_queue.put(task)
        
        logger.info(f"Task {task_id} added to queue with {len(texts)} texts and {len(images)} images")
        return task_id
    
    async def process_b64_images(self, b64_images: List[str]) -> List[Image.Image]:
        """
        Convert base64 encoded images to PIL Images.
        
        Args:
            b64_images: List of base64 encoded image strings
            
        Returns:
            List of PIL Images
        """
        if not b64_images:
            return []
        
        # For large batches or large images, run in thread pool
        if len(b64_images) > 10 or any(len(img) > 1024*1024 for img in b64_images):  # > 1MB
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._process_b64_images_sync, b64_images)
        else:
            # Small images can be processed synchronously
            return self._process_b64_images_sync(b64_images)

    def _process_b64_images_sync(self, b64_images: List[str]) -> List[Image.Image]:
        """
        Synchronous base64 image processing.
        
        Args:
            b64_images: List of base64 encoded image strings
            
        Returns:
            List of PIL Images
        """
        images = []
        for b64_image in b64_images:
            try:
                image_data = base64.b64decode(b64_image)
                image = Image.open(io.BytesIO(image_data))
                images.append(image)
            except Exception as e:
                logger.error(f"Error decoding base64 image: {e}")
                raise e
        return images
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status and results of a task.
        
        Args:
            task_id: Unique identifier for the task
            
        Returns:
            Dictionary containing task status and results, or None if not found
        """
        with self.result_lock:
            return self.result_store.get(task_id)
    
    def get_queue_size(self) -> int:
        """Get the current queue size."""
        return self.task_queue.qsize()
    
    def cleanup_completed_tasks(self, max_age_seconds: int = 3600):
        """
        Clean up old completed tasks from result store.
        
        Args:
            max_age_seconds: Maximum age in seconds for completed tasks to keep
        """
        current_time = time.time()
        tasks_to_remove = []
        
        with self.result_lock:
            for task_id, task_data in self.result_store.items():
                if task_data['status'] == 'completed' or task_data['status'] == 'failed':
                    # Use completed_at if available, otherwise fall back to created_at
                    completion_time = task_data.get('completed_at', task_data.get('created_at', current_time))
                    task_age = current_time - completion_time
                    if task_age > max_age_seconds:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.result_store[task_id]
        
        if tasks_to_remove:
            logger.info(f"Cleaned up {len(tasks_to_remove)} completed/failed tasks older than {max_age_seconds} seconds")
    
    def get_queue_stats(self) -> Dict[str, int]:
        """
        Get comprehensive queue statistics.
        
        Returns:
            Dictionary containing queue statistics
        """
        with self.result_lock:
            stats = {
                'queue_size': self.task_queue.qsize(),
                'queued_tasks': 0,
                'processing_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'total_tasks': len(self.result_store),
                'max_concurrent_requests': self.max_concurrent_requests,
                'available_gpu_slots': self.gpu_semaphore._value,
                'active_worker_threads': len([t for t in self.worker_threads if t.is_alive()])
            }
            
            for task_data in self.result_store.values():
                status = task_data['status']
                if status == 'queued':
                    stats['queued_tasks'] += 1
                elif status == 'processing':
                    stats['processing_tasks'] += 1
                elif status == 'completed':
                    stats['completed_tasks'] += 1
                elif status == 'failed':
                    stats['failed_tasks'] += 1
            
            return stats
    
    def shutdown(self):
        """Gracefully shutdown the service."""
        logger.info(f"Shutting down {self.__class__.__name__}...")
        self.shutdown_event.set()
        self.is_processing = False
        
        # Wait for all worker threads to finish
        for i, worker_thread in enumerate(self.worker_threads):
            if worker_thread.is_alive():
                logger.info(f"Waiting for worker thread {i} to finish...")
                worker_thread.join(timeout=5)
                if worker_thread.is_alive():
                    logger.warning(f"Worker thread {i} did not finish within timeout")
        
        self.worker_threads.clear()
        logger.info(f"{self.__class__.__name__} shutdown complete")
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        if hasattr(self, 'shutdown_event') and not self.shutdown_event.is_set():
            self.shutdown()
