"""
Search Router
API endpoints for anime search (API_ENDPOINTS.md compliant)
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import uuid

from app.models.schemas import (
    # New API schemas
    TextSearchRequest,
    VisualSearchRequest,
    TemporalSearchRequest,
    FilterSearchRequest,
    SearchResponse,
    ClusterResult,
    ImageItem,
    RephraseRequest,
    RephraseResponse,
    ErrorResponse,
    # Legacy schemas for backward compatibility
    SearchRequest,
    FrameResult,
    LegacyTemporalSearchRequest,
    LegacyTemporalSearchResponse,
    TemporalPair
)
from app.services.search import search_service
from app.services.translation import query_refinement_service
from app.core.milvus import milvus_client
from app.core.elastic import elastic_client

logger = logging.getLogger(__name__)
router = APIRouter()


# ========================================================================
# NEW API ENDPOINTS (API_ENDPOINTS.md compliant)
# ========================================================================

@router.post("/text", response_model=SearchResponse)
async def text_search(request: TextSearchRequest):
    """
    Text-based search using Reciprocal Rank Fusion (RRF)
    
    Combines semantic (vector) and keyword search for optimal results.
    
    - **text**: Search query text (required)
    - **mode**: Clustering mode - "moment", "timeline", "video" (default: "moment")
    - **collection**: Collection/index to search (default: "frames")
    - **top_k**: Number of results (1-1000, default: 256)
    - **state_id**: Optional state ID for follow-up searches
    
    **RRF Algorithm:**
    - Step 1: Get top_k*2 results from Milvus (vector search)
    - Step 2: Get top_k*2 results from Elasticsearch (keyword search)
    - Step 3: Apply RRF: Score = Σ 1/(60 + rank) for each source
    - Step 4: Sort by RRF score and return top_k results
    """
    try:
        logger.info(f"Text search: '{request.text}' (mode={request.mode}, top_k={request.top_k})")
        
        if not request.text or not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text query is required"
            )
        
        response = search_service.search_by_text(request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visual", response_model=SearchResponse)
async def visual_search(
    file: UploadFile = File(...),
    mode: str = Form("moment"),
    collection: str = Form("frames"),
    state_id: Optional[str] = Form(None)
):
    """
    Visual/Image-based search using uploaded image
    
    - **file**: Image file (JPG, PNG, etc.)
    - **mode**: Clustering mode - "moment", "timeline", "video" (default: "moment")
    - **collection**: Collection to search (default: "frames")
    - **state_id**: Optional state ID for follow-up searches
    """
    try:
        logger.info(f"Visual search: file={file.filename} (mode={mode})")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read image bytes
        image_data = await file.read()
        
        # Create request object
        request = VisualSearchRequest(
            mode=mode,
            collection=collection,
            state_id=state_id
        )
        
        response = search_service.search_by_image(image_data, request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visual search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/temporal", response_model=SearchResponse)
async def temporal_search(request: TemporalSearchRequest):
    """
    Temporal search for before/now/after scene sequences
    
    Find sequences of scenes that match temporal queries.
    
    - **before**: Query for scene before the main event (optional)
    - **now**: Query for the main event (optional)
    - **after**: Query for scene after the main event (optional)
    - **mode**: Clustering mode (default: "moment")
    - **top_k**: Number of sequences (default: 10)
    
    **Example:**
    ```json
    {
        "before": {"text": "character draws sword"},
        "now": {"text": "slash attack"},
        "after": {"text": "enemy falls"},
        "top_k": 10
    }
    ```
    """
    try:
        logger.info(f"Temporal search request")
        
        # At least one temporal query required
        if not request.before and not request.now and not request.after:
            raise HTTPException(
                status_code=400,
                detail="At least one of before, now, or after query is required"
            )
        
        response = search_service.search_temporal(request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Temporal search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/filter", response_model=SearchResponse)
async def filter_search(request: FilterSearchRequest):
    """
    Filter-based search with optional text query
    
    - **text**: Optional text query
    - **filters**: Filter conditions (ocr, genre, etc.)
    - **mode**: Clustering mode (default: "moment")
    - **top_k**: Number of results (default: 256)
    
    **Example:**
    ```json
    {
        "text": "fight scene",
        "filters": {
            "ocr": "Attack",
            "genre": "Action"
        },
        "top_k": 100
    }
    ```
    """
    try:
        logger.info(f"Filter search: text='{request.text}', filters={request.filters}")
        
        response = search_service.search_with_filters(request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Filter search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rephrase", response_model=RephraseResponse)
async def rephrase_query(request: RephraseRequest):
    """
    Refine/rephrase query for better CLIP-based search results.
    
    - **text**: Original query text (can be Vietnamese or English)
    - **target_lang**: Target language (default: "en") - kept for backward compatibility
    
    Uses QueryRefinementService to transform queries into detailed visual descriptions
    optimized for CLIP-based semantic search.
    
    **Example:**
    - Input: "Luffy đánh nhau"
    - Output: "Monkey D. Luffy in intense combat scene, throwing powerful punch, 
               straw hat flying, dynamic action pose, anime fight sequence"
    """
    try:
        logger.info(f"Rephrase/refine request: '{request.text}'")
        
        if not request.text or not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text is required"
            )
        
        # refine() now returns List[str] with up to 3 variants
        variants = query_refinement_service.refine(request.text)
        
        return RephraseResponse(
            status="success",
            original=request.text,
            variants=variants if variants else []
        )
        
    except Exception as e:
        logger.error(f"Rephrase/refine failed: {e}")
        return RephraseResponse(
            status="error",
            original=request.text,
            variants=[request.text]  # Return original on failure
        )


# ========================================================================
# LEGACY ENDPOINTS (Backward Compatibility)
# ========================================================================

@router.post("/search")
async def legacy_search_anime(request: SearchRequest):
    """
    [LEGACY] Search anime by image, text, or both
    
    Maintained for backward compatibility. Use /text, /visual, or /filter instead.
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
        
        # Perform search using legacy methods
        if has_image and has_text:
            # Hybrid search - use text search with RRF
            logger.info("Performing hybrid search...")
            text_req = TextSearchRequest(
                text=request.text_query,
                top_k=request.top_k
            )
            response = search_service.search_by_text(text_req)
        elif has_image:
            # Image search only
            logger.info("Performing image search...")
            response = search_service.legacy_search_by_image(
                image_input=image_input,
                top_k=request.top_k,
                filters=request.filters
            )
            # Convert to SearchResponse format
            return response
        else:
            # Text search only with RRF
            logger.info("Performing text search...")
            text_req = TextSearchRequest(
                text=request.text_query,
                top_k=request.top_k
            )
            response = search_service.search_by_text(text_req)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/upload")
async def legacy_search_by_upload(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    text_query: Optional[str] = Form(None)
):
    """
    [LEGACY] Search anime by uploading image
    
    Maintained for backward compatibility. Use /visual instead.
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
        
        # Perform search using new visual search
        request = VisualSearchRequest(mode="moment")
        response = search_service.search_by_image(image_bytes, request)
        
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


@router.post("/search/temporal")
async def legacy_temporal_search(request: LegacyTemporalSearchRequest):
    """
    [LEGACY] Temporal search for action sequences
    
    Maintained for backward compatibility. Use /temporal instead.
    """
    try:
        logger.info(f"Legacy temporal search: '{request.previous_action}' -> '{request.current_action}'")
        
        # Validate input
        if not request.current_action or not request.previous_action:
            raise HTTPException(
                status_code=400,
                detail="Both current_action and previous_action are required"
            )
        
        # Use legacy temporal search
        response = await search_service.legacy_search_temporal(request)
        
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
async def translate_text(text: str = Query(...)):
    """
    [LEGACY] Translate Vietnamese text to English
    
    Use /rephrase instead for new implementations.
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
