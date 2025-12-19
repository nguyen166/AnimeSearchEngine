"""
Search Service
Implements Pure Vector Search using Milvus (CLIP Embeddings)
Elasticsearch is used only for metadata enrichment, not for ranking.
"""

import logging
import time
import uuid
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from app.core.milvus import milvus_client
from app.core.elastic import elastic_client
from app.services.embedding import embedding_service
from app.services.translation import query_refinement_service
from app.models.schemas import (
    TextSearchRequest,
    VisualSearchRequest,
    TemporalSearchRequest,
    FilterSearchRequest,
    SearchResponse,
    ClusterResult,
    ImageItem,
    ErrorResponse,
    # Legacy models for backward compatibility
    FrameResult,
    LegacyTemporalSearchRequest,
    LegacyTemporalSearchResponse,
    TemporalPair
)
from app.config import settings

logger = logging.getLogger(__name__)


class SearchService:
    """
    Search service implementing Pure Vector Search using CLIP embeddings.
    Results are ranked solely by vector similarity from Milvus.
    """
    
    # ========================================================================
    # Main Search Methods (API_ENDPOINTS.md compliant)
    # ========================================================================
    
    @staticmethod
    def search_by_text(request: TextSearchRequest) -> SearchResponse:
        """
        Text search using Pure Vector Search (CLIP Embeddings)
        
        Results are ranked solely by vector similarity from Milvus.
        Elasticsearch is used only for metadata enrichment.
        
        Args:
            request: TextSearchRequest with text, mode, collection, top_k, state_id
            
        Returns:
            SearchResponse matching API_ENDPOINTS.md specification
        """
        start_time = time.time()
        
        try:
            text_query = request.text
            top_k = request.top_k
            mode = request.mode
            
            logger.info(f"Starting vector search: '{text_query}' (top_k={top_k})")
            
            # Step 1: Generate embedding from text
            logger.info("Encoding text query...")
            query_embedding = embedding_service.encode_text(text_query)
            
            # Step 2: Search Milvus (vector similarity)
            logger.info(f"Searching Milvus for top {top_k} results...")
            raw_results = milvus_client.search(
                query_vectors=[query_embedding],
                top_k=top_k,
                filters=None
            )[0]  # Single query
            
            logger.info(f"Milvus returned {len(raw_results)} results")
            
            # Step 3: Enrich with metadata from Elasticsearch
            enriched_results = SearchService._enrich_vector_results(raw_results)
            
            # Step 4: Format as (id, data) tuples for clustering
            sorted_results = [(r['id'], r) for r in enriched_results]
            
            # Step 5: Convert to API response format
            clusters = SearchService._format_results_as_clusters(sorted_results, mode)
            
            # Generate state_id for follow-up searches
            state_id = str(uuid.uuid4())
            
            processing_time = time.time() - start_time
            logger.info(f"Vector search completed in {processing_time:.3f}s")
            
            return SearchResponse(
                status="success",
                state_id=state_id,
                mode=mode,
                results=clusters,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Text search failed: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return SearchResponse(
                status="error",
                state_id=None,
                mode=request.mode,
                results=[],
                processing_time=processing_time
            )
    
    @staticmethod
    def search_by_image(
        image_data: bytes,
        request: VisualSearchRequest
    ) -> SearchResponse:
        """
        Visual/Image search using vector similarity (LEGACY - single image)
        
        Args:
            image_data: Raw image bytes
            request: VisualSearchRequest with mode, collection, state_id
            
        Returns:
            SearchResponse matching API_ENDPOINTS.md specification
        """
        start_time = time.time()
        
        try:
            mode = request.mode
            top_k = 32  # Default for visual search
            
            logger.info(f"Starting visual search (top_k={top_k})")
            
            # Step 1: Generate embedding from image
            logger.info("Encoding query image...")
            query_embedding = embedding_service.encode_image_bytes(image_data)
            
            # Step 2: Search Milvus
            logger.info(f"Searching Milvus for top {top_k} results...")
            vector_results = milvus_client.search(
                query_vectors=[query_embedding],
                top_k=top_k,
                filters=None
            )[0]
            
            # Step 3: Enrich with metadata and format
            enriched_results = SearchService._enrich_vector_results(vector_results)
            
            # Step 4: Convert to clusters
            clusters = SearchService._format_results_as_clusters(
                [(r['id'], r) for r in enriched_results],
                mode
            )
            
            state_id = str(uuid.uuid4())
            processing_time = time.time() - start_time
            
            return SearchResponse(
                status="success",
                state_id=state_id,
                mode=mode,
                results=clusters,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Visual search failed: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return SearchResponse(
                status="error",
                state_id=None,
                mode=request.mode,
                results=[],
                processing_time=processing_time
            )

    @staticmethod
    def search_visual(
        collection: str,
        mode: str,
        text: Optional[str] = None,
        base64_images: List[str] = []
    ) -> SearchResponse:
        """
        Visual/Multimodal search - supports multiple images + optional text
        
        Args:
            collection: Collection to search
            mode: Clustering mode (moment, video, timeline)
            text: Optional text query for multimodal search
            base64_images: List of base64 encoded images
            
        Returns:
            SearchResponse matching API_ENDPOINTS.md specification
        """
        start_time = time.time()
        
        try:
            top_k = 256  # Default for visual search
            
            logger.info(f"Starting visual search: {len(base64_images)} images, text='{text}' (top_k={top_k})")
            
            if not base64_images and not text:
                return SearchResponse(
                    status="error",
                    state_id=None,
                    mode=mode,
                    results=[],
                    processing_time=time.time() - start_time
                )
            
            # Generate embeddings
            embeddings = []
            
            # Text embedding
            if text:
                logger.info("Encoding text query...")
                text_embedding = embedding_service.encode_text(text)
                embeddings.append(text_embedding)
            
            # Image embeddings
            for idx, b64_img in enumerate(base64_images):
                logger.info(f"Encoding image {idx + 1}/{len(base64_images)}...")
                img_embedding = embedding_service.encode_image_base64(b64_img)
                embeddings.append(img_embedding)
            
            # Average embeddings if multiple
            if len(embeddings) > 1:
                # Manual averaging without numpy
                num_dims = len(embeddings[0])
                query_embedding = [
                    sum(emb[i] for emb in embeddings) / len(embeddings)
                    for i in range(num_dims)
                ]
            else:
                query_embedding = embeddings[0]
            
            # Search Milvus
            logger.info(f"Searching Milvus for top {top_k} results...")
            vector_results = milvus_client.search(
                query_vectors=[query_embedding],
                top_k=top_k,
                filters=None
            )[0]
            
            # Enrich with metadata and format
            enriched_results = SearchService._enrich_vector_results(vector_results)
            
            # Convert to clusters
            clusters = SearchService._format_results_as_clusters(
                [(r['id'], r) for r in enriched_results],
                mode
            )
            
            state_id = str(uuid.uuid4())
            processing_time = time.time() - start_time
            
            return SearchResponse(
                status="success",
                state_id=state_id,
                mode=mode,
                results=clusters,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Visual search failed: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return SearchResponse(
                status="error",
                state_id=None,
                mode=mode,
                results=[],
                processing_time=processing_time
            )
    
    @staticmethod
    def search_temporal(request: TemporalSearchRequest) -> SearchResponse:
        """
        Temporal search for before/now/after scene sequences with Temporal Alignment.
        """
        start_time = time.time()
        
        try:
            # Lấy top_k lớn hơn một chút để có đủ dữ liệu cho việc khớp (alignment)
            search_limit = request.top_k * 3 
            
            # Mặc định cửa sổ thời gian (nếu request không có thì dùng 120s)
            time_window = getattr(request, 'time_window', 120) 
            mode = request.mode if hasattr(request, 'mode') else "moment"
            
            # Extract text queries
            before_text = request.before.text if request.before else None
            now_text = request.now.text if request.now else None
            after_text = request.after.text if request.after else None
            
            if not now_text:
                raise ValueError("Query 'Now' is required for temporal search")

            logger.info(f"Temporal search: before='{before_text}', now='{now_text}', after='{after_text}'")
            
            # 1. Thực hiện tìm kiếm độc lập (Independent Search)
            
            # Search NOW (Bắt buộc)
            now_emb = embedding_service.encode_text(now_text)
            raw_now = milvus_client.search([now_emb], top_k=search_limit)[0]
            results_now = SearchService._enrich_vector_results(raw_now)
            
            # Search BEFORE (Optional)
            results_before = []
            if before_text:
                before_emb = embedding_service.encode_text(before_text)
                raw_before = milvus_client.search([before_emb], top_k=search_limit)[0]
                results_before = SearchService._enrich_vector_results(raw_before)
                
            # Search AFTER (Optional)
            results_after = []
            if after_text:
                after_emb = embedding_service.encode_text(after_text)
                raw_after = milvus_client.search([after_emb], top_k=search_limit)[0]
                results_after = SearchService._enrich_vector_results(raw_after)

            # 2. Temporal Alignment (Khớp thời gian)
            aligned_clusters = SearchService._align_temporal_results(
                results_now, 
                results_before, 
                results_after, 
                time_window=time_window,
                top_k=request.top_k
            )
            
            state_id = str(uuid.uuid4())
            processing_time = time.time() - start_time
            
            return SearchResponse(
                status="success",
                state_id=state_id,
                mode=mode,
                results=aligned_clusters,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Temporal search failed: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return SearchResponse(
                status="error",
                state_id=None,
                mode="moment",
                results=[],
                processing_time=processing_time
            )

    @staticmethod
    def _align_temporal_results(
        results_now: List[Dict],
        results_before: List[Dict],
        results_after: List[Dict],
        time_window: int,
        top_k: int
    ) -> List[ClusterResult]:
        """
        Core logic: Align Independent search results by Video ID and Timestamp.
        Strategy: Min-Score Selection & Ascending Sort (Lower score is better).
        """
        
        # Tối ưu hóa: Gom nhóm Before và After theo anime_id để tìm kiếm nhanh
        before_map = defaultdict(list)
        for r in results_before:
            before_map[r['anime_id']].append(r)
            
        after_map = defaultdict(list)
        for r in results_after:
            after_map[r['anime_id']].append(r)
            
        final_sequences = []
        
        for now_item in results_now:
            anime_id = now_item['anime_id']
            now_ts = now_item['timestamp']
            current_score = now_item['score']
            
            sequence_items = []
            
            # --- 1. Find Best BEFORE ---
            best_before = None
            if results_before:
                candidates = before_map.get(anime_id, [])
                # Khởi tạo score là vô cùng lớn để tìm MIN
                best_match_score = float('inf')
                
                for cand in candidates:
                    ts = cand['timestamp']
                    # Điều kiện: Cùng video, Trước 'Now', Trong khoảng time_window
                    if 0 <= (now_ts - ts) <= time_window:
                        # Logic: Chọn điểm THẤP NHẤT (Lower is Better)
                        if cand['score'] < best_match_score:
                            best_match_score = cand['score']
                            best_before = cand
                
                # Nếu có query Before mà không tìm thấy -> Bỏ qua
                if not best_before:
                    continue
            
            # --- 2. Find Best AFTER ---
            best_after = None
            if results_after:
                candidates = after_map.get(anime_id, [])
                best_match_score = float('inf')
                
                for cand in candidates:
                    ts = cand['timestamp']
                    # Điều kiện: Sau 'Now', Trong khoảng time_window
                    if 0 <= (ts - now_ts) <= time_window:
                        # Logic: Chọn điểm THẤP NHẤT
                        if cand['score'] < best_match_score:
                            best_match_score = cand['score']
                            best_after = cand
                            
                if not best_after:
                    continue

            # --- 3. Construct Sequence ---
            
            # Add Before Item
            if best_before:
                img = SearchService._convert_dict_to_image_item(best_before)
                img.temporalPosition = 'before'
                sequence_items.append(img)
                current_score += best_before['score']
            
            # Add Now Item
            img_now = SearchService._convert_dict_to_image_item(now_item)
            img_now.temporalPosition = 'now'
            sequence_items.append(img_now)
            
            # Add After Item
            if best_after:
                img_after = SearchService._convert_dict_to_image_item(best_after)
                img_after.temporalPosition = 'after'
                sequence_items.append(img_after)
                current_score += best_after['score']
            
            # Tính điểm trung bình cho cả chuỗi
            avg_score = current_score / len(sequence_items)
            
            final_sequences.append({
                "score": avg_score,
                "items": sequence_items,
                "video_url": now_item.get('source_url') or now_item.get('video_url')
            })
            
        # Sắp xếp Giảm DẦN (Descending) theo yêu cầu của bạn
        # Top 1 sẽ là chuỗi có Score thấp nhất
        final_sequences.sort(key=lambda x: x['score'], reverse=True)
        
        # Convert to ClusterResult format
        clusters = []
        for idx, seq in enumerate(final_sequences[:top_k]):
            clusters.append(ClusterResult(
                cluster_name=f"Sequence {idx + 1} (Score: {seq['score']:.2f})",
                url=seq['video_url'],
                image_list=seq['items']
            ))
            
        return clusters

    @staticmethod
    def _convert_dict_to_image_item(data: Dict) -> ImageItem:
        """Helper to convert dictionary data to ImageItem model"""
        return ImageItem(
            id=data['id'],
            path=f"/static/{data.get('frame_path', '')}",
            score=data.get('score', 0),
            time_in_seconds=data.get('timestamp', 0.0),
            name=f"Frame at {data.get('timestamp', 0):.1f}s",
            videoId=data.get('anime_id', ''),
            videoName=data.get('anime_title', '')
        )
    
    @staticmethod
    def search_with_filters(request: FilterSearchRequest) -> SearchResponse:
        """
        Filter-based search with optional text query using Pure Vector Search
        
        Args:
            request: FilterSearchRequest with filters (ocr, genre) and optional text
            
        Returns:
            SearchResponse matching API_ENDPOINTS.md specification
        """
        start_time = time.time()
        
        try:
            top_k = request.top_k
            mode = request.mode
            text_query = request.text
            filters = request.filters
            
            logger.info(f"Filter search: text='{text_query}', filters={filters}")
            
            # Build filter criteria for post-filtering
            filter_criteria = {}
            if filters:
                if filters.ocr:
                    filter_criteria['ocr'] = filters.ocr
                if filters.genre:
                    filter_criteria['genre'] = filters.genre
            
            if text_query:
                # Vector search with text query
                logger.info("Encoding text query...")
                query_embedding = embedding_service.encode_text(text_query)
                
                # Search Milvus - fetch more if we need to filter
                fetch_limit = top_k * 3 if filter_criteria else top_k
                
                logger.info(f"Searching Milvus for top {fetch_limit} results...")
                raw_results = milvus_client.search(
                    query_vectors=[query_embedding],
                    top_k=fetch_limit,
                    filters=None
                )[0]
                
                # Enrich with metadata
                enriched_results = SearchService._enrich_vector_results(raw_results)
                
                # Apply filters in Python if needed
                if filter_criteria:
                    enriched_results = SearchService._apply_filters_to_results(
                        enriched_results, filter_criteria
                    )
                
                # Limit to top_k
                enriched_results = enriched_results[:top_k]
                sorted_results = [(r['id'], r) for r in enriched_results]
                
            else:
                # Filter-only search via Elasticsearch (no vector ranking)
                logger.info("Performing filter-only search via Elasticsearch...")
                es_filters = {}
                if filters:
                    if filters.ocr:
                        es_filters['ocr'] = filters.ocr
                    if filters.genre:
                        es_filters['genre'] = filters.genre
                        
                keyword_results = SearchService._search_elasticsearch_metadata(
                    "*", top_k, es_filters
                )
                sorted_results = [(r['id'], r) for r in keyword_results]
            
            clusters = SearchService._format_results_as_clusters(sorted_results, mode)
            
            state_id = str(uuid.uuid4())
            processing_time = time.time() - start_time
            
            return SearchResponse(
                status="success",
                state_id=state_id,
                mode=mode,
                results=clusters,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Filter search failed: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return SearchResponse(
                status="error",
                state_id=None,
                mode=request.mode,
                results=[],
                processing_time=processing_time
            )
    
    # ========================================================================
    # Helper Methods for Vector Search
    # ========================================================================
    
    @staticmethod
    def _apply_filters_to_results(
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply filters to search results in Python
        
        Args:
            results: List of enriched search results
            filters: Dictionary of filter criteria (ocr, genre, etc.)
            
        Returns:
            Filtered list of results
        """
        if not filters:
            return results
        
        filtered = []
        for r in results:
            match = True
            
            # OCR filter
            if 'ocr' in filters and filters['ocr']:
                ocr_text = r.get('ocr', '') or ''
                if not any(term.lower() in ocr_text.lower() for term in filters['ocr']):
                    match = False
            
            # Genre filter
            if 'genre' in filters and filters['genre']:
                genres = r.get('genres', []) or []
                if not any(g in genres for g in filters['genre']):
                    match = False
            
            if match:
                filtered.append(r)
        
        return filtered
    
    @staticmethod
    def _search_elasticsearch_metadata(
        text_query: str,
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search Elasticsearch for metadata lookup only (not for ranking)
        
        Args:
            text_query: Text query or "*" for all
            limit: Maximum number of results
            filters: Optional filters (genre, ocr, etc.)
            
        Returns:
            List of results with metadata
        """
        try:
            # Search Elasticsearch
            raw_results = elastic_client.search(
                query=text_query,
                filters=filters,
                size=limit
            )
            
            # Flatten and normalize results
            normalized = []
            for doc in raw_results:
                # Handle both flat documents and nested frame structures
                if 'frames' in doc:
                    # Nested structure: expand frames
                    for frame in doc.get('frames', []):
                        normalized.append({
                            'id': frame.get('id', frame.get('frame_id', '')),
                            'anime_id': doc.get('anime_id', ''),
                            'anime_title': doc.get('title', ''),
                            'episode': frame.get('episode', 1),
                            'timestamp': frame.get('timestamp', 0.0),
                            'frame_path': frame.get('frame_path', ''),
                            'score': 1.0,  # No ranking score for metadata lookup
                            'video_url': doc.get('video_url', ''),
                            'source_url': doc.get('source_url', '')
                        })
                else:
                    # Flat structure
                    normalized.append({
                        'id': doc.get('id', doc.get('frame_id', '')),
                        'anime_id': doc.get('anime_id', ''),
                        'anime_title': doc.get('title', doc.get('anime_title', '')),
                        'episode': doc.get('episode', 1),
                        'timestamp': doc.get('timestamp', 0.0),
                        'frame_path': doc.get('frame_path', ''),
                        'score': 1.0,  # No ranking score for metadata lookup
                        'video_url': doc.get('video_url', ''),
                        'source_url': doc.get('source_url', '')
                    })
            
            return normalized
            
        except Exception as e:
            logger.error(f"Elasticsearch metadata search failed: {e}")
            return []
    
    # ========================================================================
    # Other Helper Methods
    # ========================================================================
    
    @staticmethod
    def _refine_query(text: str) -> str:
        """
        Refine query for better CLIP embedding performance.
        
        Uses QueryRefinementService to transform raw queries into detailed
        visual descriptions optimized for semantic search.
        
        Args:
            text: Raw user query (can be Vietnamese or English)
            
        Returns:
            Refined English query with visual descriptions (first variant)
        """
        try:
            logger.info(f"Refining query: '{text}'")
            # refine() returns List[str], take first variant for embedding
            variants = query_refinement_service.refine(text)
            refined = variants[0] if variants else text
            logger.info(f"Refined to: '{refined}'")
            return refined
        except Exception as e:
            logger.warning(f"Query refinement failed, using original: {e}")
            return text
    
    @staticmethod
    def _is_vietnamese(text: str) -> bool:
        """Detect if text contains Vietnamese characters"""
        vietnamese_chars = [
            'à', 'á', 'ả', 'ã', 'ạ', 'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ',
            'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'đ', 'è', 'é', 'ẻ', 'ẽ',
            'ẹ', 'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ì', 'í', 'ỉ', 'ĩ',
            'ị', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'ô', 'ồ', 'ố', 'ổ', 'ỗ',
            'ộ', 'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ù', 'ú', 'ủ', 'ũ',
            'ụ', 'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ'
        ]
        return any(char in text.lower() for char in vietnamese_chars)
    
    @staticmethod
    def _enrich_vector_results(
        vector_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich Milvus results with metadata from Elasticsearch
        
        Args:
            vector_results: Raw results from Milvus
            
        Returns:
            Enriched results with full metadata
        """
        enriched = []
        
        for vr in vector_results:
            doc_id = vr.get('id', '')
            anime_id = vr.get('anime_id', '')
            
            # Try to get metadata from Elasticsearch
            try:
                metadata = elastic_client.get_document(doc_id)
            except Exception:
                metadata = None
            
            # Convert distance to similarity score
            # distance = vr.get('score', vr.get('distance', 0))
            # # For COSINE distance: similarity = 1 - distance
            # # For L2 distance: similarity = 1 / (1 + distance)
            # similarity = max(0, 1 - distance) if distance <= 1 else 1 / (1 + distance)
            similarity = vr.get('score', 0.0)
            
            
            enriched.append({
                'id': doc_id,
                'anime_id': anime_id,
                'anime_title': metadata.get('title', '') if metadata else vr.get('anime_title', ''),
                'episode': vr.get('episode', 1),
                'timestamp': vr.get('timestamp', 0.0),
                'frame_path': metadata.get('frame_path', '') if metadata else vr.get('frame_path', ''),
                'score': similarity,
                'title': metadata.get('title', '') if metadata else '',
                'description': metadata.get('description', '') if metadata else '',
                'source_url': metadata.get('source_url', '') if metadata else '',
                'video_url': metadata.get('video_url', '') if metadata else '',
                
            })
        
        return enriched
    
    @staticmethod
    def _format_results_as_clusters(
        sorted_results: List[Tuple[str, Dict[str, Any]]],
        mode: str
    ) -> List[ClusterResult]:
        """
        Format results into ClusterResult structure for API response
        
        Groups results by anime_id/episode for 'moment' mode
        
        Args:
            sorted_results: List of (doc_id, doc_data) tuples sorted by score
            mode: Clustering mode (moment, timeline, video, etc.)
            
        Returns:
            List of ClusterResult objects
        """
        if mode == "video":
            # Group by anime + episode
            groups = defaultdict(list)
            group_urls = {}  # Store video_url for each group
            
            for doc_id, doc_data in sorted_results:
                anime_id = doc_data.get('anime_id', 'unknown')
                episode = doc_data.get('episode', 0)
                anime_title = doc_data.get('anime_title', doc_data.get('title', anime_id))
                # Prefer source_url over video_url as it's more commonly populated
                video_url = doc_data.get('source_url', '') or doc_data.get('video_url', '')
                
                group_key = f"{anime_title} - Episode {episode}"
                
                # Store video_url for this group (first non-empty wins)
                if group_key not in group_urls or not group_urls[group_key]:
                    group_urls[group_key] = video_url
                
                # Get frame_path from document or generate from timestamp
                frame_path = doc_data.get('frame_path', '')
                if not frame_path:
                    # Fallback: generate path from timestamp
                    timestamp = doc_data.get('timestamp', 0.0)
                    frame_path = f"frames/{anime_id}/ep{episode:03d}/frame_{timestamp:.2f}.jpg"
                
                # Create ImageItem
                image_item = ImageItem(
                    id=doc_id,
                    path=f"/static/{frame_path}",
                    score=doc_data.get('score', 0),
                    time_in_seconds=doc_data.get('timestamp', 0.0),
                    name=f"Frame at {doc_data.get('timestamp', 0):.1f}s",
                    videoId=anime_id,
                    videoName=anime_title,
                    frameNumber=int(doc_data.get('timestamp', 0) * 24)  # Approximate frame number
                )
                
                groups[group_key].append(image_item)
            
            # Convert to ClusterResult list
            # Sort items within each cluster by score descending
            clusters = []
            for name, items in groups.items():
                # Sort images by score descending
                sorted_items = sorted(items, key=lambda x: x.score or 0, reverse=True)
                clusters.append(
                    ClusterResult(
                        cluster_name=name,
                        url=group_urls.get(name) or None,
                        image_list=sorted_items
                    )
                )
            
            # Sort clusters by their best score (first item's score) descending
            clusters.sort(key=lambda c: c.image_list[0].score if c.image_list else 0, reverse=True)
            
        else:
            # Default: single cluster with all results
            items = []
            first_url = None
            for doc_id, doc_data in sorted_results:
                # Capture first video_url for the cluster (prefer source_url)
                if first_url is None:
                    first_url = doc_data.get('source_url', '') or doc_data.get('video_url', '')
                
                # Get frame_path from document or generate from timestamp
                frame_path = doc_data.get('frame_path', '')
                if not frame_path:
                    # Fallback: generate path from timestamp
                    anime_id = doc_data.get('anime_id', 'unknown')
                    episode = doc_data.get('episode', 0)
                    timestamp = doc_data.get('timestamp', 0.0)
                    frame_path = f"frames/{anime_id}/ep{episode:03d}/frame_{timestamp:.2f}.jpg"
                
                items.append(ImageItem(
                    id=doc_id,
                    path=f"/static/{frame_path}",
                    score=doc_data.get('score', 0),
                    time_in_seconds=doc_data.get('timestamp', 0.0),
                    name=f"Frame at {doc_data.get('timestamp', 0):.1f}s",
                    videoId=doc_data.get('anime_id', ''),
                    videoName=doc_data.get('anime_title', '')
                ))
            
            # Sort items by score descending
            items.sort(key=lambda x: x.score or 0, reverse=True)
            
            clusters = [
                ClusterResult(
                    cluster_name="Search Results",
                    url=first_url or None,
                    image_list=items
                )
            ]
        
        return clusters
    
    @staticmethod
    def _combine_temporal_results(
        results_map: Dict[str, List[ClusterResult]],
        top_k: int
    ) -> List[ClusterResult]:
        """
        Combine before/now/after results into temporal sequences
        
        Args:
            results_map: Dict with 'before', 'now', 'after' keys containing results
            top_k: Maximum sequences to return
            
        Returns:
            List of ClusterResult with temporal sequences
        """
        clusters = []
        
        # Get items from each temporal position
        before_items = []
        now_items = []
        after_items = []
        
        for cluster in results_map.get('before', []):
            for item in cluster.image_list:
                item.temporalPosition = 'before'
                before_items.append(item)
        
        for cluster in results_map.get('now', []):
            for item in cluster.image_list:
                item.temporalPosition = 'now'
                now_items.append(item)
        
        for cluster in results_map.get('after', []):
            for item in cluster.image_list:
                item.temporalPosition = 'after'
                after_items.append(item)
        
        # Create sequences (simple approach: zip top items from each)
        max_sequences = min(top_k, len(now_items) if now_items else top_k)
        
        for i in range(max_sequences):
            sequence_items = []
            
            if before_items and i < len(before_items):
                sequence_items.append(before_items[i])
            
            if now_items and i < len(now_items):
                sequence_items.append(now_items[i])
            
            if after_items and i < len(after_items):
                sequence_items.append(after_items[i])
            
            if sequence_items:
                clusters.append(ClusterResult(
                    cluster_name=f"Sequence {i + 1}",
                    url=None,
                    image_list=sequence_items
                ))
        
        return clusters
    
    # ========================================================================
    # Legacy Methods (Backward Compatibility)
    # ========================================================================
    
    @staticmethod
    def legacy_search_by_image(
        image_input: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Legacy image search (backward compatibility)"""
        start_time = time.time()
        
        try:
            query_embedding = embedding_service.encode_image(image_input)
            
            filter_expr = None
            if filters:
                filter_parts = []
                if "anime_id" in filters:
                    filter_parts.append(f"anime_id == '{filters['anime_id']}'")
                if "episode" in filters:
                    filter_parts.append(f"episode == {filters['episode']}")
                if filter_parts:
                    filter_expr = " && ".join(filter_parts)
            
            vector_results = milvus_client.search(
                query_vectors=[query_embedding],
                top_k=top_k * 2,
                filters=filter_expr
            )[0]
            
            enriched = SearchService._enrich_vector_results(vector_results)
            
            results = [
                FrameResult(
                    frame_id=r['id'],
                    anime_id=r['anime_id'],
                    anime_title=r.get('anime_title', ''),
                    episode=r['episode'],
                    timestamp=r['timestamp'],
                    score=r['score'],
                    frame_path=r.get('frame_path'),
                    thumbnail_url=None
                )
                for r in enriched
            ][:top_k]
            
            return {
                "success": True,
                "query_type": "image",
                "total_results": len(results),
                "results": results,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Legacy image search failed: {e}")
            return {
                "success": False,
                "query_type": "image",
                "total_results": 0,
                "results": [],
                "processing_time": time.time() - start_time
            }
    
    @staticmethod
    def legacy_search_by_text(
        text_query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        semantic_weight: float = 0.5
    ) -> Dict[str, Any]:
        """Legacy text search using weighted sum (backward compatibility)"""
        request = TextSearchRequest(text=text_query, top_k=top_k)
        response = SearchService.search_by_text(request)
        
        # Convert to legacy format
        results = []
        for cluster in response.results:
            for item in cluster.image_list:
                results.append(FrameResult(
                    frame_id=item.id,
                    anime_id=item.videoId or '',
                    anime_title=item.videoName or '',
                    episode=1,
                    timestamp=item.time_in_seconds or 0,
                    score=item.score or 0,
                    frame_path=item.path,
                    thumbnail_url=None
                ))
        
        return {
            "success": response.status == "success",
            "query_type": "text_hybrid",
            "total_results": len(results),
            "results": results,
            "processing_time": response.processing_time or 0
        }
    
    @staticmethod
    async def legacy_search_temporal(
        request: LegacyTemporalSearchRequest,
        auto_translate: bool = True
    ) -> LegacyTemporalSearchResponse:
        """Legacy temporal search (backward compatibility)"""
        start_time = time.time()
        
        try:
            current_action = request.current_action
            previous_action = request.previous_action
            
            if auto_translate:
                if SearchService._is_vietnamese(current_action):
                    current_action = query_refinement_service.translate(current_action)
                if SearchService._is_vietnamese(previous_action):
                    previous_action = query_refinement_service.translate(previous_action)
            
            # Search for both actions
            current_results = SearchService._search_milvus(current_action, request.top_k * 5)
            previous_results = SearchService._search_milvus(previous_action, request.top_k * 5)
            
            # Pair results
            pairs = SearchService._pair_temporal_frames_legacy(
                current_results, previous_results,
                request.time_window,
                current_action, previous_action
            )
            
            pairs.sort(key=lambda x: x.combined_score, reverse=True)
            final_pairs = pairs[:request.top_k]
            
            return LegacyTemporalSearchResponse(
                success=True,
                query_type="temporal",
                total_results=len(final_pairs),
                pairs=final_pairs,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Legacy temporal search failed: {e}")
            return LegacyTemporalSearchResponse(
                success=False,
                query_type="temporal",
                total_results=0,
                pairs=[],
                processing_time=time.time() - start_time
            )
    
    @staticmethod
    def _pair_temporal_frames_legacy(
        current_results: List[Dict],
        previous_results: List[Dict],
        time_window: int,
        current_action: str,
        previous_action: str
    ) -> List[TemporalPair]:
        """Legacy temporal frame pairing"""
        pairs = []
        
        prev_index = defaultdict(list)
        for prev in previous_results:
            key = (prev.get('anime_id'), prev.get('episode'))
            prev_index[key].append(prev)
        
        for current in current_results:
            key = (current.get('anime_id'), current.get('episode'))
            
            if key not in prev_index:
                continue
            
            for prev in prev_index[key]:
                time_diff = current.get('timestamp', 0) - prev.get('timestamp', 0)
                
                if 0 < time_diff <= time_window:
                    combined_score = 0.6 * current.get('score', 0) + 0.4 * prev.get('score', 0)
                    
                    pairs.append(TemporalPair(
                        previous_frame=FrameResult(
                            frame_id=prev.get('id', ''),
                            anime_id=prev.get('anime_id', ''),
                            anime_title=prev.get('anime_title', ''),
                            episode=prev.get('episode', 1),
                            timestamp=prev.get('timestamp', 0),
                            score=prev.get('score', 0),
                            frame_path=prev.get('frame_path'),
                            thumbnail_url=None
                        ),
                        current_frame=FrameResult(
                            frame_id=current.get('id', ''),
                            anime_id=current.get('anime_id', ''),
                            anime_title=current.get('anime_title', ''),
                            episode=current.get('episode', 1),
                            timestamp=current.get('timestamp', 0),
                            score=current.get('score', 0),
                            frame_path=current.get('frame_path'),
                            thumbnail_url=None
                        ),
                        time_difference=time_diff,
                        combined_score=combined_score,
                        sequence_context=f"{previous_action} -> {current_action}"
                    ))
        
        return pairs


# Create singleton instance
search_service = SearchService()
