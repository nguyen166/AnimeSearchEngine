"""
Pydantic Schemas
Định nghĩa các models cho Input/Output của API
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request schema cho image search"""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL của hình ảnh")
    text_query: Optional[str] = Field(None, description="Text query để tìm kiếm")
    top_k: int = Field(10, ge=1, le=100, description="Số lượng kết quả trả về")
    filters: Optional[dict] = Field(None, description="Filters cho metadata (genre, year, etc.)")
    semantic_weight: float = Field(0.5, ge=0.0, le=1.0, description="Trọng số cho semantic search trong text search (0-1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAUA...",
                "top_k": 10,
                "filters": {
                    "genre": "action",
                    "year": 2023
                }
            }
        }


class FrameResult(BaseModel):
    """Thông tin về một frame tìm được"""
    frame_id: str = Field(..., description="ID của frame")
    anime_id: str = Field(..., description="ID của anime")
    anime_title: str = Field(..., description="Tên anime")
    episode: int = Field(..., description="Số tập")
    timestamp: float = Field(..., description="Thời điểm trong video (giây)")
    score: float = Field(..., description="Độ tương đồng (0-1)")
    frame_path: Optional[str] = Field(None, description="Đường dẫn đến frame")
    thumbnail_url: Optional[str] = Field(None, description="URL thumbnail")
    
    class Config:
        json_schema_extra = {
            "example": {
                "frame_id": "anime_001_ep01_00123",
                "anime_id": "anime_001",
                "anime_title": "One Piece",
                "episode": 1,
                "timestamp": 123.45,
                "score": 0.95,
                "frame_path": "/data/frames/anime_001/ep01/frame_00123.jpg"
            }
        }


class AnimeMetadata(BaseModel):
    """Metadata của anime"""
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "anime_id": "anime_001",
                "title": "One Piece",
                "title_english": "One Piece",
                "title_japanese": "ワンピース",
                "title_vietnamese": "Vua Hải Tặc",
                "genres": ["Action", "Adventure", "Fantasy"],
                "year": 1999,
                "episodes": 1000,
                "rating": 8.7,
                "studio": "Toei Animation",
                "studios": ["Toei Animation"],
                "season": "1999-Spring"
            }
        }


class SearchResponse(BaseModel):
    """Response schema cho search results"""
    success: bool = Field(True, description="Trạng thái request")
    query_type: str = Field(..., description="Loại query: image, text, hybrid")
    total_results: int = Field(..., description="Tổng số kết quả tìm được")
    results: List[FrameResult] = Field([], description="Danh sách kết quả")
    processing_time: float = Field(..., description="Thời gian xử lý (giây)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query_type": "image",
                "total_results": 10,
                "results": [],
                "processing_time": 0.156
            }
        }


class ErrorResponse(BaseModel):
    """Response schema cho errors"""
    success: bool = Field(False)
    error: str = Field(..., description="Mô tả lỗi")
    detail: Optional[str] = Field(None, description="Chi tiết lỗi")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid request",
                "detail": "Image or text query is required"
            }
        }


class IngestRequest(BaseModel):
    """Request schema cho data ingestion"""
    video_path: str = Field(..., description="Đường dẫn đến video")
    anime_id: str = Field(..., description="ID của anime")
    episode: int = Field(..., description="Số tập")
    fps: float = Field(1.0, description="Số frame mỗi giây để extract")
    metadata: Optional[AnimeMetadata] = Field(None, description="Metadata của anime")


class IngestResponse(BaseModel):
    """Response schema cho data ingestion"""
    success: bool = Field(True)
    anime_id: str
    episode: int
    frames_extracted: int = Field(..., description="Số frame đã extract")
    frames_indexed: int = Field(..., description="Số frame đã index")
    processing_time: float


class TemporalSearchRequest(BaseModel):
    """Request schema cho temporal search (tìm kiếm theo chuỗi thời gian)"""
    current_action: str = Field(..., description="Mô tả hành động chính/kết quả (ví dụ: 'explosion', 'character falls')")
    previous_action: str = Field(..., description="Mô tả hành động xảy ra trước đó/nguyên nhân (ví dụ: 'sword attack', 'character jumps')")
    time_window: int = Field(10, ge=1, le=60, description="Khoảng thời gian tối đa (giây) giữa hai hành động")
    top_k: int = Field(10, ge=1, le=100, description="Số lượng kết quả trả về")
    filters: Optional[dict] = Field(None, description="Filters cho metadata (genre, year, anime_id, etc.)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "current_action": "explosion",
                "previous_action": "character draws sword",
                "time_window": 10,
                "top_k": 10,
                "filters": {
                    "anime_id": "anime_001",
                    "genres": ["Action"]
                }
            }
        }


class TemporalPair(BaseModel):
    """Thông tin về một cặp frame trong chuỗi temporal"""
    previous_frame: FrameResult = Field(..., description="Frame của hành động trước")
    current_frame: FrameResult = Field(..., description="Frame của hành động sau")
    time_difference: float = Field(..., description="Khoảng thời gian giữa hai frame (giây)")
    combined_score: float = Field(..., description="Điểm tổng hợp của cặp")
    sequence_context: str = Field(..., description="Mô tả chuỗi hành động")


class TemporalSearchResponse(BaseModel):
    """Response schema cho temporal search results"""
    success: bool = Field(True, description="Trạng thái request")
    query_type: str = Field("temporal", description="Loại query: temporal")
    total_results: int = Field(..., description="Tổng số cặp tìm được")
    pairs: List[TemporalPair] = Field([], description="Danh sách các cặp hành động")
    processing_time: float = Field(..., description="Thời gian xử lý (giây)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query_type": "temporal",
                "total_results": 5,
                "pairs": [],
                "processing_time": 0.456
            }
        }
