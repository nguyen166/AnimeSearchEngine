"""
Search Router
API endpoints cho tìm kiếm anime
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from app.models.schemas import (
    SearchRequest,
    SearchResponse,
    ErrorResponse,
    TemporalSearchRequest,
    TemporalSearchResponse
)
from app.services.search import SearchService
from app.services.translation import translation_service
from app.core.milvus import milvus_client
from app.core.elastic import elastic_client

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_anime(request: SearchRequest):
    """
    Tìm kiếm anime bằng hình ảnh, text, hoặc cả hai
    
    - **image_base64**: Base64 encoded image (optional)
    - **image_url**: URL của hình ảnh (optional)
    - **text_query**: Text query (optional)
    - **top_k**: Số lượng kết quả (1-100)
    - **filters**: Filters cho metadata như genre, year, etc.
    """
    try:
        # Validate input
        if not request.image_base64 and not request.image_url and not request.text_query:
            raise HTTPException(
                status_code=400,
                detail="At least one of image_base64, image_url, or text_query is required"
            )
        
        # Determine search type
        has_image = bool(request.image_base64 or request.image_url)
        has_text = bool(request.text_query)
        
        # Get image input
        image_input = request.image_base64 or request.image_url
        
        # Perform search
        if has_image and has_text:
            # Hybrid search
            logger.info("Performing hybrid search...")
            response = SearchService.hybrid_search(
                image_input=image_input,
                text_query=request.text_query,
                top_k=request.top_k,
                filters=request.filters
            )
        elif has_image:
            # Image search only
            logger.info("Performing image search...")
            response = SearchService.search_by_image(
                image_input=image_input,
                top_k=request.top_k,
                filters=request.filters
            )
        else:
            # Text search only
            logger.info("Performing text search...")
            response = SearchService.search_by_text(
                text_query=request.text_query,
                top_k=request.top_k,
                filters=request.filters,
                semantic_weight=request.semantic_weight
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/upload", response_model=SearchResponse)
async def search_by_upload(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    text_query: Optional[str] = Form(None)
):
    """
    Tìm kiếm anime bằng cách upload hình ảnh
    
    - **file**: Image file (JPG, PNG, etc.)
    - **top_k**: Số lượng kết quả
    - **text_query**: Optional text query
    """
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Perform search
        if text_query:
            response = SearchService.hybrid_search(
                image_input=image_bytes,
                text_query=text_query,
                top_k=top_k
            )
        else:
            response = SearchService.search_by_image(
                image_input=image_bytes,
                top_k=top_k
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anime/{anime_id}")
async def get_anime_details(anime_id: str):
    """
    Lấy thông tin chi tiết của một anime
    
    - **anime_id**: ID của anime
    """
    try:
        anime_doc = elastic_client.get_document(anime_id)
        
        if not anime_doc:
            raise HTTPException(
                status_code=404,
                detail=f"Anime {anime_id} not found"
            )
        
        return {
            "success": True,
            "data": anime_doc
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get anime details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anime")
async def list_anime(
    limit: int = 20,
    offset: int = 0,
    genre: Optional[str] = None,
    year: Optional[int] = None
):
    """
    Liệt kê danh sách anime với pagination và filters
    
    - **limit**: Số lượng kết quả mỗi page
    - **offset**: Offset cho pagination
    - **genre**: Filter theo genre
    - **year**: Filter theo năm
    """
    try:
        filters = {}
        if genre:
            filters["genres"] = genre
        if year:
            filters["year"] = year
        
        results = elastic_client.search(
            filters=filters,
            size=limit
        )
        
        return {
            "success": True,
            "total": len(results),
            "limit": limit,
            "offset": offset,
            "data": results
        }
        
    except Exception as e:
        logger.error(f"Failed to list anime: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/temporal", response_model=TemporalSearchResponse)
async def temporal_search(request: TemporalSearchRequest):
    """
    Tìm kiếm chuỗi hành động theo thời gian (Temporal Search)
    
    Tính năng này cho phép tìm kiếm các chuỗi hành động nhân quả trong Anime,
    ví dụ: "Nhân vật rút kiếm" → "Cảnh nổ lớn" trong khoảng thời gian 10 giây.
    
    - **current_action**: Mô tả hành động chính/kết quả (bắt buộc)
    - **previous_action**: Mô tả hành động xảy ra trước đó/nguyên nhân (bắt buộc)
    - **time_window**: Khoảng thời gian tối đa (giây) giữa hai hành động (1-60, mặc định 10)
    - **top_k**: Số lượng kết quả trả về (1-100, mặc định 10)
    - **filters**: Filters cho metadata (anime_id, genre, year, etc.)
    
    **Example Request:**
    ```json
    {
        "current_action": "explosion",
        "previous_action": "character draws sword",
        "time_window": 10,
        "top_k": 10
    }
    ```
    
    **Logic:**
    1. Tìm kiếm song song cả previous và current actions
    2. Ghép cặp các frame thỏa mãn:
       - Cùng anime_id và episode
       - Previous timestamp < current timestamp
       - Khoảng cách thời gian <= time_window
    3. Tính điểm tổng hợp và ranking
    4. Trả về top_k cặp có điểm cao nhất
    """
    try:
        logger.info(f"Temporal search request: '{request.previous_action}' -> '{request.current_action}'")
        
        # Validate input
        if not request.current_action or not request.previous_action:
            raise HTTPException(
                status_code=400,
                detail="Both current_action and previous_action are required"
            )
        
        # Perform temporal search
        response = await SearchService.search_temporal(request)
        
        if not response.success:
            logger.warning("Temporal search returned no results or encountered errors")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Temporal search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Temporal search failed: {str(e)}"
        )


@router.post("/translate")
async def translate_text(text: str):
    """
    Dịch văn bản từ tiếng Việt sang tiếng Anh
    
    Endpoint này sử dụng Google Gemini (hoặc fallback provider) để dịch thuật.
    
    - **text**: Văn bản tiếng Việt cần dịch
    
    **Example:**
    ```
    POST /translate?text=Nhân vật rút kiếm
    ```
    """
    try:
        if not text or not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text is required"
            )
        
        translated = translation_service.translate(text)
        
        return {
            "success": True,
            "original": text,
            "translated": translated,
            "mode": translation_service.mode
        }
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )


@router.get("/stats")
async def get_system_stats():
    """
    Lấy thống kê hệ thống (số lượng anime, frames, translation, etc.)
    """
    try:
        milvus_stats = milvus_client.get_stats()
        elastic_stats = elastic_client.get_stats()
        translation_stats = translation_service.get_stats()
        
        return {
            "success": True,
            "milvus": milvus_stats,
            "elasticsearch": elastic_stats,
            "translation": translation_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Kiểm tra health của các services
    """
    try:
        status = {
            "api": "healthy",
            "milvus": "unknown",
            "elasticsearch": "unknown"
        }
        
        # Check Milvus
        try:
            milvus_client.get_stats()
            status["milvus"] = "healthy"
        except:
            status["milvus"] = "unhealthy"
        
        # Check Elasticsearch
        try:
            elastic_client.get_stats()
            status["elasticsearch"] = "healthy"
        except:
            status["elasticsearch"] = "unhealthy"
        
        # Overall status
        overall_healthy = all(
            v == "healthy" 
            for k, v in status.items()
        )
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "services": status
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )
