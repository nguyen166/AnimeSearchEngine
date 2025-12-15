"""
Search Service
Logic nghiệp vụ cho tìm kiếm anime
"""

import logging
import time
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from app.core.milvus import milvus_client
from app.core.elastic import elastic_client
from app.services.embedding import embedding_service
from app.services.translation import translation_service
from app.models.schemas import (
    SearchRequest, 
    SearchResponse, 
    FrameResult,
    TemporalSearchRequest,
    TemporalSearchResponse,
    TemporalPair
)
from app.config import settings

logger = logging.getLogger(__name__)


class SearchService:
    """Service để xử lý logic tìm kiếm"""
    
    @staticmethod
    def search_by_image(
        image_input: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """
        Tìm kiếm anime bằng hình ảnh
        
        Args:
            image_input: Base64 string hoặc URL
            top_k: Số lượng kết quả
            filters: Filters cho metadata
            
        Returns:
            SearchResponse
        """
        start_time = time.time()
        
        try:
            # 1. Tạo embedding từ hình ảnh
            logger.info("Encoding query image...")
            query_embedding = embedding_service.encode_image(image_input)
            
            # 2. Tìm kiếm trong Milvus
            logger.info(f"Searching Milvus for top {top_k} results...")
            
            # Build filter expression nếu có
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
                top_k=top_k * 2,  # Lấy nhiều hơn để filter
                filters=filter_expr
            )[0]  # Chỉ có 1 query
            
            # 3. Lấy metadata từ Elasticsearch và kết hợp kết quả
            logger.info("Fetching metadata from Elasticsearch...")
            results = SearchService._enrich_results(vector_results, filters)
            
            # 4. Filter theo threshold
            results = [
                r for r in results 
                if r.score >= settings.SIMILARITY_THRESHOLD
            ][:top_k]
            
            processing_time = time.time() - start_time
            
            return SearchResponse(
                success=True,
                query_type="image",
                total_results=len(results),
                results=results,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            processing_time = time.time() - start_time
            return SearchResponse(
                success=False,
                query_type="image",
                total_results=0,
                results=[],
                processing_time=processing_time
            )
    
    @staticmethod
    def search_by_text(
        text_query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        semantic_weight: float = 0.5
    ) -> SearchResponse:
        """
        Tìm kiếm anime bằng text với hybrid search (Semantic + Keyword)
        
        Args:
            text_query: Text query
            top_k: Số lượng kết quả
            filters: Filters cho metadata
            semantic_weight: Trọng số cho semantic search (0-1), keyword weight = 1 - semantic_weight
            
        Returns:
            SearchResponse
        """
        start_time = time.time()
        
        try:
            results = []
            
            # Step 1: Translation - Translate to English for better CLIP performance
            logger.info(f"Translating query: {text_query}")
            try:
                translated_query = translation_service.translate(
                    text_query,
                    target_lang='en'
                )
                logger.info(f"Translated query: {translated_query}")
            except Exception as e:
                logger.warning(f"Translation failed, using original query: {e}")
                translated_query = text_query
            
            # Step 2: Embedding - Generate text embedding
            logger.info("Encoding text query to embedding...")
            query_embedding = embedding_service.encode_text(translated_query)
            
            # Step 3: Milvus Search - Semantic search via vector similarity
            logger.info(f"Performing semantic search in Milvus (top {top_k * 2})...")
            
            # Build filter expression nếu có
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
                top_k=top_k * 2,  # Lấy nhiều hơn để merge
                filters=filter_expr
            )[0]  # Chỉ có 1 query
            
            # Enrich với metadata từ Elasticsearch
            semantic_results = SearchService._enrich_results(vector_results, filters)
            
            # Áp dụng semantic weight
            for r in semantic_results:
                r.score *= semantic_weight
            
            results.extend(semantic_results)
            
            # Step 4: Elasticsearch Search - Keyword search
            logger.info(f"Performing keyword search in Elasticsearch with query: {text_query}")
            
            anime_results = elastic_client.search(
                query=text_query,
                filters=filters,
                size=top_k * 2
            )
            
            # Convert sang FrameResult format
            keyword_weight = 1.0 - semantic_weight
            for anime in anime_results:
                # Lấy frame đầu tiên của mỗi anime
                frames = anime.get("frames", [])
                if frames:
                    frame = frames[0]
                    results.append(FrameResult(
                        frame_id=frame.get("frame_id", ""),
                        anime_id=anime.get("anime_id", ""),
                        anime_title=anime.get("title", ""),
                        episode=frame.get("episode", 1),
                        timestamp=frame.get("timestamp", 0.0),
                        score=1.0 * keyword_weight,  # Apply keyword weight
                        frame_path=frame.get("frame_path"),
                        thumbnail_url=None
                    ))
            
            # Step 5: Merge Results - Combine semantic + keyword results
            logger.info("Merging semantic and keyword results...")
            
            # Group by frame_id và sum scores
            merged = {}
            for r in results:
                if r.frame_id in merged:
                    merged[r.frame_id].score += r.score
                else:
                    merged[r.frame_id] = r
            
            # Sort by combined score
            final_results = sorted(
                merged.values(),
                key=lambda x: x.score,
                reverse=True
            )[:top_k]
            
            processing_time = time.time() - start_time
            
            logger.info(f"Text search completed: {len(final_results)} results in {processing_time:.2f}s")
            
            return SearchResponse(
                success=True,
                query_type="text_hybrid",
                total_results=len(final_results),
                results=final_results,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            processing_time = time.time() - start_time
            return SearchResponse(
                success=False,
                query_type="text_hybrid",
                total_results=0,
                results=[],
                processing_time=processing_time
            )
    
    @staticmethod
    def hybrid_search(
        image_input: Optional[str] = None,
        text_query: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        image_weight: float = 0.7
    ) -> SearchResponse:
        """
        Tìm kiếm kết hợp image và text
        
        Args:
            image_input: Base64 string hoặc URL
            text_query: Text query
            top_k: Số lượng kết quả
            filters: Filters cho metadata
            image_weight: Trọng số cho image search (0-1)
            
        Returns:
            SearchResponse
        """
        start_time = time.time()
        
        try:
            results = []
            
            # 1. Image search
            if image_input:
                logger.info("Performing image search...")
                img_response = SearchService.search_by_image(
                    image_input, top_k * 2, filters
                )
                if img_response.success:
                    # Áp dụng trọng số
                    for r in img_response.results:
                        r.score *= image_weight
                    results.extend(img_response.results)
            
            # 2. Text search
            if text_query:
                logger.info("Performing text search...")
                text_response = SearchService.search_by_text(
                    text_query, top_k * 2, filters
                )
                if text_response.success:
                    # Áp dụng trọng số
                    text_weight = 1.0 - image_weight
                    for r in text_response.results:
                        r.score *= text_weight
                    results.extend(text_response.results)
            
            # 3. Merge và rank kết quả
            # Group by frame_id và sum scores
            merged = {}
            for r in results:
                if r.frame_id in merged:
                    merged[r.frame_id].score += r.score
                else:
                    merged[r.frame_id] = r
            
            # Sort by score
            final_results = sorted(
                merged.values(),
                key=lambda x: x.score,
                reverse=True
            )[:top_k]
            
            processing_time = time.time() - start_time
            
            return SearchResponse(
                success=True,
                query_type="hybrid",
                total_results=len(final_results),
                results=final_results,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            processing_time = time.time() - start_time
            return SearchResponse(
                success=False,
                query_type="hybrid",
                total_results=0,
                results=[],
                processing_time=processing_time
            )
    
    @staticmethod
    def _enrich_results(
        vector_results: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None
    ) -> List[FrameResult]:
        """
        Kết hợp kết quả từ Milvus với metadata từ Elasticsearch
        
        Args:
            vector_results: Kết quả từ Milvus
            filters: Filters để apply thêm
            
        Returns:
            List of enriched FrameResult
        """
        enriched = []
        
        for vr in vector_results:
            anime_id = vr.get("anime_id")
            
            # Lấy metadata từ Elasticsearch
            anime_doc = elastic_client.get_document(anime_id)
            
            if anime_doc:
                # Apply additional filters nếu có
                if filters:
                    # Filter by genre
                    if "genres" in filters:
                        required_genres = filters["genres"]
                        if not any(g in anime_doc.get("genres", []) for g in required_genres):
                            continue
                    
                    # Filter by year
                    if "year" in filters:
                        if anime_doc.get("year") != filters["year"]:
                            continue
                
                # Convert distance sang similarity score (L2 distance)
                # Score càng cao càng tốt (1 - normalized_distance)
                distance = vr.get("score", 0)
                similarity = max(0, 1 - distance / 2)  # Normalize
                
                enriched.append(FrameResult(
                    frame_id=vr.get("id", ""),
                    anime_id=anime_id,
                    anime_title=anime_doc.get("title", "Unknown"),
                    episode=vr.get("episode", 0),
                    timestamp=vr.get("timestamp", 0.0),
                    score=similarity,
                    frame_path=None,  # TODO: Build frame path
                    thumbnail_url=None
                ))
        
        return enriched
    
    @staticmethod
    async def search_temporal(
        request: TemporalSearchRequest,
        auto_translate: bool = True
    ) -> TemporalSearchResponse:
        """
        Tìm kiếm chuỗi hành động theo thời gian (Temporal Search)
        
        Args:
            request: TemporalSearchRequest với current_action, previous_action, time_window
            auto_translate: Tự động dịch sang tiếng Anh nếu phát hiện tiếng Việt
            
        Returns:
            TemporalSearchResponse với danh sách các cặp hành động
        """
        start_time = time.time()
        
        try:
            # Detect and translate Vietnamese to English if needed
            current_action = request.current_action
            previous_action = request.previous_action
            
            if auto_translate:
                # Simple detection: if contains Vietnamese characters
                if SearchService._is_vietnamese(current_action):
                    logger.info(f"Translating current_action from Vietnamese: {current_action}")
                    current_action = translation_service.translate(current_action)
                    logger.info(f"Translated to: {current_action}")
                
                if SearchService._is_vietnamese(previous_action):
                    logger.info(f"Translating previous_action from Vietnamese: {previous_action}")
                    previous_action = translation_service.translate(previous_action)
                    logger.info(f"Translated to: {previous_action}")
            
            logger.info(f"Starting temporal search: '{previous_action}' -> '{current_action}'")
            
            # Step 1: Parallel Search cho cả current và previous actions
            # Lấy nhiều kết quả hơn để có không gian lọc
            search_multiplier = 5
            search_limit = request.top_k * search_multiplier
            
            logger.info(f"Performing parallel text search (limit={search_limit} each)...")
            
            # Sử dụng asyncio.gather để chạy song song
            current_search_task = asyncio.create_task(
                SearchService._async_text_search(
                    text_query=current_action,
                    top_k=search_limit,
                    filters=request.filters
                )
            )
            
            previous_search_task = asyncio.create_task(
                SearchService._async_text_search(
                    text_query=previous_action,
                    top_k=search_limit,
                    filters=request.filters
                )
            )
            
            # Đợi cả hai kết quả
            current_results, previous_results = await asyncio.gather(
                current_search_task,
                previous_search_task
            )
            
            logger.info(f"Found {len(current_results)} current frames, {len(previous_results)} previous frames")
            
            # Step 2: Pairing Algorithm - Ghép cặp các frame thỏa mãn điều kiện temporal
            temporal_pairs = SearchService._pair_temporal_frames(
                current_results=current_results,
                previous_results=previous_results,
                time_window=request.time_window,
                current_action=current_action,
                previous_action=previous_action
            )
            
            logger.info(f"Found {len(temporal_pairs)} valid temporal pairs")
            
            # Step 3: Scoring & Ranking
            # Sắp xếp theo điểm tổng hợp giảm dần
            temporal_pairs.sort(key=lambda x: x.combined_score, reverse=True)
            
            # Lấy top_k kết quả
            final_pairs = temporal_pairs[:request.top_k]
            
            processing_time = time.time() - start_time
            logger.info(f"Temporal search completed in {processing_time:.3f}s")
            
            return TemporalSearchResponse(
                success=True,
                query_type="temporal",
                total_results=len(final_pairs),
                pairs=final_pairs,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Temporal search failed: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return TemporalSearchResponse(
                success=False,
                query_type="temporal",
                total_results=0,
                pairs=[],
                processing_time=processing_time
            )
    
    @staticmethod
    async def _async_text_search(
        text_query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[FrameResult]:
        """
        Async wrapper cho text search để sử dụng trong parallel processing
        
        Args:
            text_query: Text query
            top_k: Số lượng kết quả
            filters: Filters cho metadata
            
        Returns:
            List of FrameResult
        """
        try:
            # Tìm kiếm trong Elasticsearch
            anime_results = elastic_client.search(
                query=text_query,
                filters=filters,
                size=top_k
            )
            
            # Convert sang FrameResult format và flatten tất cả frames
            results = []
            for anime in anime_results:
                frames = anime.get("frames", [])
                for frame in frames:
                    results.append(FrameResult(
                        frame_id=frame.get("frame_id", ""),
                        anime_id=anime.get("anime_id", ""),
                        anime_title=anime.get("title", ""),
                        episode=frame.get("episode", 1),
                        timestamp=frame.get("timestamp", 0.0),
                        score=0.8,  # Placeholder score from ES
                        frame_path=frame.get("frame_path"),
                        thumbnail_url=None
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Async text search failed: {e}")
            return []
    
    @staticmethod
    def _pair_temporal_frames(
        current_results: List[FrameResult],
        previous_results: List[FrameResult],
        time_window: int,
        current_action: str,
        previous_action: str
    ) -> List[TemporalPair]:
        """
        Ghép cặp các frame thỏa mãn điều kiện temporal
        
        Logic:
        - Với mỗi current frame, tìm previous frame cùng anime_id, episode
        - Previous timestamp phải < current timestamp
        - Khoảng cách thời gian <= time_window
        - Chọn previous frame có điểm cao nhất nếu có nhiều ứng viên
        
        Args:
            current_results: Danh sách frame của hành động hiện tại
            previous_results: Danh sách frame của hành động trước đó
            time_window: Khoảng thời gian tối đa (giây)
            current_action: Mô tả hành động hiện tại
            previous_action: Mô tả hành động trước đó
            
        Returns:
            List of TemporalPair
        """
        pairs = []
        
        # Index previous results theo (anime_id, episode) để tìm kiếm nhanh
        prev_index: Dict[Tuple[str, int], List[FrameResult]] = {}
        for prev_frame in previous_results:
            key = (prev_frame.anime_id, prev_frame.episode)
            if key not in prev_index:
                prev_index[key] = []
            prev_index[key].append(prev_frame)
        
        # Duyệt qua từng current frame
        for current_frame in current_results:
            key = (current_frame.anime_id, current_frame.episode)
            
            # Tìm các previous frame cùng anime và episode
            if key not in prev_index:
                continue
            
            candidate_prevs = prev_index[key]
            
            # Lọc các previous frame thỏa mãn điều kiện temporal
            valid_prevs = []
            for prev_frame in candidate_prevs:
                time_diff = current_frame.timestamp - prev_frame.timestamp
                
                # Kiểm tra điều kiện:
                # 1. Previous phải xảy ra trước current (time_diff > 0)
                # 2. Khoảng cách <= time_window
                if 0 < time_diff <= time_window:
                    valid_prevs.append((prev_frame, time_diff))
            
            # Nếu có ứng viên, chọn cái có điểm cao nhất
            if valid_prevs:
                # Sắp xếp theo điểm giảm dần
                valid_prevs.sort(key=lambda x: x[0].score, reverse=True)
                best_prev, time_diff = valid_prevs[0]
                
                # Tính điểm tổng hợp (trung bình có trọng số)
                # Ưu tiên current frame hơn (weight=0.6)
                combined_score = (0.6 * current_frame.score + 0.4 * best_prev.score)
                
                # Bonus điểm nếu khoảng cách thời gian ngắn
                time_bonus = 1.0 - (time_diff / time_window) * 0.1
                combined_score *= time_bonus
                
                # Tạo context mô tả chuỗi
                sequence_context = (
                    f"Sequence: [{previous_action}] "
                    f"({best_prev.timestamp:.1f}s) → "
                    f"[{current_action}] "
                    f"({current_frame.timestamp:.1f}s) | "
                    f"Δt={time_diff:.1f}s"
                )
                
                # Tạo TemporalPair
                pair = TemporalPair(
                    previous_frame=best_prev,
                    current_frame=current_frame,
                    time_difference=time_diff,
                    combined_score=combined_score,
                    sequence_context=sequence_context
                )
                
                pairs.append(pair)
        
        return pairs
    
    @staticmethod
    def _is_vietnamese(text: str) -> bool:
        """
        Phát hiện xem text có chứa ký tự tiếng Việt không
        
        Args:
            text: Text cần kiểm tra
            
        Returns:
            True nếu có ký tự tiếng Việt
        """
        # Vietnamese characters with diacritics
        vietnamese_chars = [
            'à', 'á', 'ả', 'ã', 'ạ', 'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ',
            'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'đ', 'è', 'é', 'ẻ', 'ẽ',
            'ẹ', 'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ì', 'í', 'ỉ', 'ĩ',
            'ị', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'ô', 'ồ', 'ố', 'ổ', 'ỗ',
            'ộ', 'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ù', 'ú', 'ủ', 'ũ',
            'ụ', 'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ'
        ]
        
        text_lower = text.lower()
        return any(char in text_lower for char in vietnamese_chars)


# Create instance
search_service = SearchService()
