import os
from enum import Enum
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class HealthStatus(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Service status")

class EmbeddingRequest(BaseModel):
    """
    Represents an embedding request.
    """
    model: str = Field(..., description="The model to use for embedding.")
    texts: List[str] = Field(default=[], description="The texts to embed.")
    b64_images: List[str] = Field(default=[], description="The base64 encoded images to embed.")
    priority: Priority = Field(default=Priority.NORMAL, description="Task priority")

class Embedding(BaseModel):
    """
    Represents an embedding object in an embedding response.
    """
    embedding: List[float] = Field(..., description="The embedding vector.")
    index: int = Field(..., description="The index of the embedding in the list.")
    object: str = Field(default="embedding", description="The object type, always 'embedding'.")

class EmbeddingResponse(BaseModel):
    """
    Represents an embedding response.
    """
    object: str = Field("list", description="The object type, always 'list'.")
    data: List[Embedding] = Field(..., description="List of embedding objects.")
    model: str = Field(..., description="The model used for embedding.")

class EmbeddingError(BaseModel):
    """Embedding error schema"""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")

class Model(BaseModel):
    """
    Represents a model in the models list response.
    """
    id: str = Field(..., description="The model ID.")
    object: str = Field("model", description="The object type, always 'model'.")
    created: int = Field(..., description="The creation timestamp.")
    owned_by: str = Field("openai", description="The owner of the model.")

class ModelsResponse(BaseModel):
    """
    Represents the response for the models list endpoint.
    """
    object: str = Field("list", description="The object type, always 'list'.")
    data: List[Model] = Field(..., description="List of models.")

class TaskInfo(BaseModel):
    """Task information schema"""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Task progress (0.0 to 1.0)")
    created_at: datetime = Field(..., description="Task creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Task processing start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    priority: Priority = Field(default=Priority.NORMAL, description="Task priority")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    queue_position: Optional[int] = Field(None, description="Current position in queue")

class TaskStatusResponse(BaseModel):
    """Task status response schema"""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Task progress (0.0 to 1.0)")
    created_at: datetime = Field(..., description="Task creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Task processing start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    queue_position: Optional[int] = Field(None, description="Current position in queue")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    error: Optional[EmbeddingError] = Field(None, description="Error information if task failed")

class TaskSubmissionResponse(BaseModel):
    """Task submission response schema"""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    queue_position: int = Field(..., description="Position in queue")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

class QueueStatsResponse(BaseModel):
    """Queue statistics response schema"""
    queue_size: int = Field(..., description="Current queue size")
    queued_tasks: int = Field(..., description="Number of queued tasks")
    processing_tasks: int = Field(..., description="Number of processing tasks")
    completed_tasks: int = Field(..., description="Number of completed tasks")
    failed_tasks: int = Field(..., description="Number of failed tasks")
    avg_processing_time: float = Field(..., description="Average processing time in seconds")
    processed_tasks: int = Field(..., description="Total processed tasks")