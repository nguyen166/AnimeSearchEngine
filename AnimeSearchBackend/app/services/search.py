"""
Search Service
Implements Reciprocal Rank Fusion (RRF) for hybrid search
combining Milvus (Vector Search) and Elasticsearch (Keyword Search)
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
from app.services.translation import translation_service
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

# RRF constant (standard value used in literature)
RRF_K = 60


class SearchService:
    """
    Search service implementing Reciprocal Rank Fusion (RRF)
    for hybrid semantic + keyword search
    """
    
    # ========================================================================
    # Main Search Methods (API_ENDPOINTS.md compliant)
    # ========================================================================
    
    @staticmethod
    def search_by_text(request: TextSearchRequest) -> SearchResponse:
        """
        Text search using Reciprocal Rank Fusion (RRF)
        
        Combines:
        - Milvus: Vector similarity search (semantic)
        - Elasticsearch: Keyword/metadata search
        
        RRF Formula: score = Σ 1/(k + rank) for each source
        
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
            
            logger.info(f"Starting RRF text search: '{text_query}' (top_k={top_k})")
            
            # Step 1: Translate to English for better embedding performance
            translated_query = SearchService._translate_query(text_query)
            
            # Step 2: Get results from both sources (fetch more for RRF merging)
            fetch_limit = top_k * 2
            
            # Step 2a: Milvus Vector Search
            logger.info(f"Performing Milvus vector search (limit={fetch_limit})...")
            vector_results = SearchService._search_milvus(translated_query, fetch_limit)
            logger.info(f"Milvus returned {len(vector_results)} results")
            
            # Step 2b: Elasticsearch Keyword Search
            logger.info(f"Performing Elasticsearch keyword search (limit={fetch_limit})...")
            keyword_results = SearchService._search_elasticsearch(text_query, fetch_limit)
            logger.info(f"Elasticsearch returned {len(keyword_results)} results")
            
            # Step 3: Apply Reciprocal Rank Fusion
            logger.info("Applying Reciprocal Rank Fusion (RRF)...")
            fused_results = SearchService._apply_rrf(
                vector_results=vector_results,
                keyword_results=keyword_results,
                k=RRF_K
            )
            
            # Step 4: Sort by RRF score and take top_k
            sorted_results = sorted(
                fused_results.items(),
                key=lambda x: x[1]['rrf_score'],
                reverse=True
            )[:top_k]
            
            logger.info(f"RRF produced {len(sorted_results)} final results")
            
            # Step 5: Convert to API response format
            clusters = SearchService._format_results_as_clusters(sorted_results, mode)
            
            # Generate state_id for follow-up searches
            state_id = str(uuid.uuid4())
            
            processing_time = time.time() - start_time
            logger.info(f"Text search completed in {processing_time:.3f}s")
            
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
        Visual/Image search using vector similarity
        
        Args:
            image_data: Raw image bytes
            request: VisualSearchRequest with mode, collection, state_id
            
        Returns:
            SearchResponse matching API_ENDPOINTS.md specification
        """
        start_time = time.time()
        
        try:
            mode = request.mode
            top_k = 256  # Default for visual search
            
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
    def search_temporal(request: TemporalSearchRequest) -> SearchResponse:
        """
        Temporal search for before/now/after scene sequences
        
        Args:
            request: TemporalSearchRequest with before, now, after queries
            
        Returns:
            SearchResponse with results grouped as scenes with 3 frames each
        """
        start_time = time.time()
        
        try:
            top_k = request.top_k
            mode = request.mode if hasattr(request, 'mode') else "moment"
            
            # Extract text queries
            before_text = request.before.text if request.before else None
            now_text = request.now.text if request.now else None
            after_text = request.after.text if request.after else None
            
            logger.info(f"Temporal search: before='{before_text}', now='{now_text}', after='{after_text}'")
            
            # Search for each temporal component
            results_map = {}
            
            if before_text:
                before_req = TextSearchRequest(text=before_text, top_k=top_k)
                before_resp = SearchService.search_by_text(before_req)
                results_map['before'] = before_resp.results
            
            if now_text:
                now_req = TextSearchRequest(text=now_text, top_k=top_k)
                now_resp = SearchService.search_by_text(now_req)
                results_map['now'] = now_resp.results
            
            if after_text:
                after_req = TextSearchRequest(text=after_text, top_k=top_k)
                after_resp = SearchService.search_by_text(after_req)
                results_map['after'] = after_resp.results
            
            # Combine results into temporal clusters
            clusters = SearchService._combine_temporal_results(results_map, top_k)
            
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
    def search_with_filters(request: FilterSearchRequest) -> SearchResponse:
        """
        Filter-based search with optional text query
        
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
            
            # Build Elasticsearch filter query
            es_filters = {}
            if filters:
                if filters.ocr:
                    es_filters['ocr'] = filters.ocr
                if filters.genre:
                    es_filters['genre'] = filters.genre
            
            # If text query provided, use RRF hybrid search with filters
            if text_query:
                # Similar to text search but with filters applied
                translated_query = SearchService._translate_query(text_query)
                
                fetch_limit = top_k * 2
                
                # Vector search (no direct filter support in Milvus, filter post-retrieval)
                vector_results = SearchService._search_milvus(translated_query, fetch_limit)
                
                # Elasticsearch search with filters
                keyword_results = SearchService._search_elasticsearch(
                    text_query, fetch_limit, es_filters
                )
                
                # Apply RRF
                fused_results = SearchService._apply_rrf(vector_results, keyword_results, RRF_K)
                
                sorted_results = sorted(
                    fused_results.items(),
                    key=lambda x: x[1]['rrf_score'],
                    reverse=True
                )[:top_k]
                
            else:
                # Filter-only search via Elasticsearch
                keyword_results = SearchService._search_elasticsearch(
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
    # RRF Implementation
    # ========================================================================
    
    @staticmethod
    def _apply_rrf(
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        k: int = 60
    ) -> Dict[str, Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion to combine results from multiple sources
        
        RRF Score = Σ 1/(k + rank) for each source where the document appears
        
        Args:
            vector_results: Results from Milvus vector search (ordered by similarity)
            keyword_results: Results from Elasticsearch keyword search (ordered by relevance)
            k: RRF constant (default=60, standard in literature)
            
        Returns:
            Dictionary mapping document_id to {document_data, rrf_score, ranks}
        """
        fused = {}
        
        # Process vector results (assign rank 1, 2, 3, ...)
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result.get('id', '')
            if not doc_id:
                continue
            
            rrf_score = 1.0 / (k + rank)
            
            if doc_id not in fused:
                fused[doc_id] = {
                    **result,
                    'rrf_score': rrf_score,
                    'vector_rank': rank,
                    'keyword_rank': None
                }
            else:
                fused[doc_id]['rrf_score'] += rrf_score
                fused[doc_id]['vector_rank'] = rank
        
        # Process keyword results (assign rank 1, 2, 3, ...)
        for rank, result in enumerate(keyword_results, start=1):
            doc_id = result.get('id', '')
            if not doc_id:
                continue
            
            rrf_score = 1.0 / (k + rank)
            
            if doc_id not in fused:
                fused[doc_id] = {
                    **result,
                    'rrf_score': rrf_score,
                    'vector_rank': None,
                    'keyword_rank': rank
                }
            else:
                fused[doc_id]['rrf_score'] += rrf_score
                fused[doc_id]['keyword_rank'] = rank
        
        logger.debug(f"RRF fused {len(fused)} unique documents")
        
        return fused
    
    # ========================================================================
    # Search Source Methods
    # ========================================================================
    
    @staticmethod
    def _search_milvus(
        text_query: str,
        limit: int,
        filters: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search Milvus using text embedding
        
        Args:
            text_query: Text query (should be in English for best results)
            limit: Maximum number of results
            filters: Optional Milvus filter expression
            
        Returns:
            List of results with id, anime_id, episode, timestamp, score, etc.
        """
        try:
            # Generate text embedding
            query_embedding = embedding_service.encode_text(text_query)
            
            # Search Milvus
            raw_results = milvus_client.search(
                query_vectors=[query_embedding],
                top_k=limit,
                filters=filters
            )[0]  # Single query
            
            # Enrich with metadata
            enriched = SearchService._enrich_vector_results(raw_results)
            
            return enriched
            
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return []
    
    @staticmethod
    def _search_elasticsearch(
        text_query: str,
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search Elasticsearch using keyword matching
        
        Args:
            text_query: Text query
            limit: Maximum number of results
            filters: Optional filters (genre, ocr, etc.)
            
        Returns:
            List of results with id, anime_id, episode, timestamp, score, etc.
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
                            'score': doc.get('_score', 1.0),
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
                        'score': doc.get('_score', 1.0),
                        'video_url': doc.get('video_url', ''),
                        'source_url': doc.get('source_url', '')
                    })
            
            return normalized
            
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    @staticmethod
    def _translate_query(text: str) -> str:
        """Translate text to English for better embedding performance"""
        try:
            if SearchService._is_vietnamese(text):
                logger.info(f"Translating Vietnamese query: {text}")
                translated = translation_service.translate(text, target_lang='en')
                logger.info(f"Translated to: {translated}")
                return translated
            return text
        except Exception as e:
            logger.warning(f"Translation failed, using original: {e}")
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
            distance = vr.get('score', vr.get('distance', 0))
            # For COSINE distance: similarity = 1 - distance
            # For L2 distance: similarity = 1 / (1 + distance)
            similarity = max(0, 1 - distance) if distance <= 1 else 1 / (1 + distance)
            
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
                'video_url': metadata.get('video_url', '') if metadata else ''
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
        if mode == "moment":
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
                
                # Create ImageItem
                image_item = ImageItem(
                    id=doc_id,
                    path=f"/{anime_id}/ep{episode:03d}/",
                    score=doc_data.get('rrf_score', doc_data.get('score', 0)),
                    time_in_seconds=doc_data.get('timestamp', 0.0),
                    name=f"Frame at {doc_data.get('timestamp', 0):.1f}s",
                    videoId=anime_id,
                    videoName=anime_title,
                    frameNumber=int(doc_data.get('timestamp', 0) * 24)  # Approximate frame number
                )
                
                groups[group_key].append(image_item)
            
            # Convert to ClusterResult list
            clusters = [
                ClusterResult(
                    cluster_name=name,
                    url=group_urls.get(name) or None,
                    image_list=items
                )
                for name, items in groups.items()
            ]
            
        else:
            # Default: single cluster with all results
            items = []
            first_url = None
            for doc_id, doc_data in sorted_results:
                # Capture first video_url for the cluster (prefer source_url)
                if first_url is None:
                    first_url = doc_data.get('source_url', '') or doc_data.get('video_url', '')
                items.append(ImageItem(
                    id=doc_id,
                    path=f"/{doc_data.get('anime_id', 'unknown')}/",
                    score=doc_data.get('rrf_score', doc_data.get('score', 0)),
                    time_in_seconds=doc_data.get('timestamp', 0.0),
                    name=f"Frame at {doc_data.get('timestamp', 0):.1f}s",
                    videoId=doc_data.get('anime_id', ''),
                    videoName=doc_data.get('anime_title', '')
                ))
            
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
                    current_action = translation_service.translate(current_action)
                if SearchService._is_vietnamese(previous_action):
                    previous_action = translation_service.translate(previous_action)
            
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
