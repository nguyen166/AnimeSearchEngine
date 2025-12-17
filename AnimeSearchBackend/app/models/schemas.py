"""
Pydantic Schemas
API Input/Output models aligned with API_ENDPOINTS.md specification
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


# ============================================================================
# Request Models
# ============================================================================

class TextSearchRequest(BaseModel):
    """
    Request schema for text search
    Endpoint: POST /search/text
    """
    text: str = Field(..., description="Search query text")
    mode: str = Field("moment", description="Clustering mode: moment, timeline, location, video")
    collection: str = Field("default", description="Model collection name")
    top_k: int = Field(256, ge=1, le=1000, description="Number of results to return")
    state_id: Optional[str] = Field(None, description="Previous search state ID for continuation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "a man walking in the rain",
                "mode": "moment",
                "collection": "default",
                "top_k": 256,
                "state_id": None
            }
        }


class VisualSearchRequest(BaseModel):
    """
    Request schema for visual/image search
    Endpoint: POST /search/visual (multipart/form-data)
    """
    mode: str = Field("moment", description="Clustering mode")
    collection: str = Field("default", description="Model collection name")
    state_id: Optional[str] = Field(None, description="Previous search state ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "mode": "moment",
                "collection": "default"
            }
        }


class TemporalQueryItem(BaseModel):
    """Temporal query component for before/now/after scenes"""
    text: Optional[str] = Field(None, description="Text description of the scene")


class TemporalSearchRequest(BaseModel):
    """
    Request schema for temporal search
    Endpoint: POST /search/visual/temporal
    """
    collection: str = Field("default", description="Model collection name")
    state_id: Optional[str] = Field(None, description="Previous search state ID")
    before: Optional[TemporalQueryItem] = Field(None, description="Scene description before")
    now: Optional[TemporalQueryItem] = Field(None, description="Main event description")
    after: Optional[TemporalQueryItem] = Field(None, description="Scene description after")
    top_k: int = Field(256, ge=1, le=1000, description="Number of results to return")
    
    class Config:
        json_schema_extra = {
            "example": {
                "collection": "default",
                "before": {"text": "character draws sword"},
                "now": {"text": "explosion"},
                "after": {"text": "character falls"}
            }
        }


class FilterQuery(BaseModel):
    """Filter criteria for metadata-based search"""
    ocr: Optional[List[str]] = Field(None, description="Text visible on screen (OCR)")
    genre: Optional[List[str]] = Field(None, description="Video genre categories")


class FilterSearchRequest(BaseModel):
    """
    Request schema for filter-based search
    Endpoint: POST /search/filter
    """
    mode: str = Field("moment", description="Clustering mode")
    filters: Optional[FilterQuery] = Field(None, description="Metadata filters")
    text: Optional[str] = Field(None, description="Optional visual description")
    top_k: int = Field(256, ge=1, le=1000, description="Number of results to return")
    collection: str = Field("default", description="Model collection name")
    state_id: Optional[str] = Field(None, description="Previous search state ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "mode": "moment",
                "filters": {
                    "ocr": ["game over"],
                    "genre": ["horror", "comedy"]
                },
                "text": "person walking",
                "top_k": 256
            }
        }


class RephraseRequest(BaseModel):
    """Request schema for rephrase suggestions"""
    text: str = Field(..., description="Original query text")
    target_lang: str = Field("en", description="Target language for translation")
    message_ref: Optional[str] = Field(None, description="Unique message reference")


# ============================================================================
# Response Models
# ============================================================================

class ImageItem(BaseModel):
    """
    Individual frame/image result item
    Matches ImageItem interface in API_ENDPOINTS.md
    """
    id: str = Field(..., description="Image identifier (e.g., 'L01_V001/001234')")
    path: str = Field(..., description="Path prefix for image URL")
    score: Optional[float] = Field(None, ge=0, le=1, description="Relevance score (0-1)")
    time_in_seconds: Optional[float] = Field(None, description="Timestamp in video")
    name: Optional[str] = Field(None, description="Display name")
    videoId: Optional[str] = Field(None, description="Video identifier")
    videoName: Optional[str] = Field(None, description="Video display name")
    frameNumber: Optional[int] = Field(None, description="Frame number in video")
    temporalPosition: Optional[Literal['before', 'now', 'after']] = Field(
        None, description="Position in temporal search"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "one-piece/ep001/001234",
                "path": "/one-piece/ep001/",
                "score": 0.95,
                "time_in_seconds": 123.45,
                "name": "Frame 1234",
                "videoId": "one-piece",
                "videoName": "One Piece",
                "frameNumber": 1234
            }
        }


class ClusterResult(BaseModel):
    """
    Cluster/group of search results
    Matches ClusterResult interface in API_ENDPOINTS.md
    """
    cluster_name: str = Field(..., description="Cluster/scene name or Anime Title")
    url: Optional[str] = Field(None, description="Optional cluster URL")
    image_list: List[ImageItem] = Field(default_factory=list, description="Frames in this cluster")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cluster_name": "One Piece - Episode 1",
                "url": None,
                "image_list": []
            }
        }


class SearchResponse(BaseModel):
    """
    Standard search response
    Matches SearchResponse interface in API_ENDPOINTS.md
    """
    status: str = Field("success", description="Response status: success or error")
    state_id: Optional[str] = Field(None, description="State ID for follow-up searches")
    mode: str = Field("moment", description="Clustering mode used")
    results: List[ClusterResult] = Field(default_factory=list, description="Search results")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "state_id": "abc123",
                "mode": "moment",
                "results": [
                    {
                        "cluster_name": "One Piece - Episode 1",
                        "url": None,
                        "image_list": [
                            {
                                "id": "one-piece/ep001/001234",
                                "path": "/one-piece/ep001/",
                                "score": 0.95,
                                "time_in_seconds": 123.45
                            }
                        ]
                    }
                ]
            }
        }


class RephraseResponse(BaseModel):
    """Response schema for rephrase suggestions"""
    status: str = Field("success", description="Response status")
    original: str = Field(..., description="Original query text")
    rephrased: str = Field(..., description="Rephrased/translated query")
    message_ref: Optional[str] = Field(None, description="Message reference from request")
    variants: List[str] = Field(default_factory=list, description="Additional rephrased variants")


class ErrorResponse(BaseModel):
    """Response schema for errors"""
    status: str = Field("error", description="Error status")
    error: str = Field(..., description="Error description")
    detail: Optional[str] = Field(None, description="Detailed error message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "error": "Invalid request",
                "detail": "Text query is required"
            }
        }


# ============================================================================
# Legacy Models (kept for backward compatibility)
# ============================================================================

class LegacySearchRequest(BaseModel):
    """Legacy request schema (deprecated, use TextSearchRequest)"""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL of image")
    text_query: Optional[str] = Field(None, description="Text query")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    filters: Optional[dict] = Field(None, description="Metadata filters")
    semantic_weight: float = Field(0.5, ge=0.0, le=1.0, description="Semantic search weight")


# Alias for backward compatibility
SearchRequest = LegacySearchRequest


class FrameResult(BaseModel):
    """Legacy frame result (use ImageItem for new implementations)"""
    frame_id: str = Field(..., description="Frame ID")
    anime_id: str = Field(..., description="Anime ID")
    anime_title: str = Field(..., description="Anime title")
    episode: int = Field(..., description="Episode number")
    timestamp: float = Field(..., description="Timestamp in video (seconds)")
    score: float = Field(..., description="Similarity score (0-1)")
    frame_path: Optional[str] = Field(None, description="Frame file path")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")


class AnimeMetadata(BaseModel):
    """Anime metadata"""
    anime_id: str
    title: str
    title_english: Optional[str] = None
    title_japanese: Optional[str] = None
    title_vietnamese: Optional[str] = None
    genres: List[str] = []
    year: Optional[int] = None
    episodes: int = 0
    rating: Optional[float] = None
    description: Optional[str] = None
    studio: Optional[str] = None
    studios: Optional[List[str]] = None
    season: Optional[str] = None


class IngestRequest(BaseModel):
    """Request schema for data ingestion"""
    video_path: str = Field(..., description="Path to video file")
    anime_id: str = Field(..., description="Anime ID")
    episode: int = Field(..., description="Episode number")
    fps: float = Field(1.0, description="Frames per second to extract")
    metadata: Optional[AnimeMetadata] = Field(None, description="Anime metadata")


class IngestResponse(BaseModel):
    """Response schema for data ingestion"""
    success: bool = Field(True)
    anime_id: str
    episode: int
    frames_extracted: int = Field(..., description="Number of frames extracted")
    frames_indexed: int = Field(..., description="Number of frames indexed")
    processing_time: float


class LegacyTemporalSearchRequest(BaseModel):
    """Legacy temporal search request"""
    current_action: str = Field(..., description="Current/result action")
    previous_action: str = Field(..., description="Previous/cause action")
    time_window: int = Field(10, ge=1, le=60, description="Max time window (seconds)")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    filters: Optional[dict] = Field(None, description="Metadata filters")


class TemporalPair(BaseModel):
    """Temporal pair result"""
    previous_frame: FrameResult
    current_frame: FrameResult
    time_difference: float
    combined_score: float
    sequence_context: str


class LegacyTemporalSearchResponse(BaseModel):
    """Legacy temporal search response"""
    success: bool = Field(True)
    query_type: str = Field("temporal")
    total_results: int
    pairs: List[TemporalPair] = []
    processing_time: float
