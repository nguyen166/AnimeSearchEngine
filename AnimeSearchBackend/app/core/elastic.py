"""
Elasticsearch Connection
Quản lý kết nối và thao tác với Elasticsearch cho metadata
"""

from typing import List, Optional, Dict, Any
import logging
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from app.config import settings

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """Singleton class để quản lý kết nối Elasticsearch"""
    
    _instance = None
    _client: Optional[Elasticsearch] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ElasticsearchClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._connect()
    
    def _connect(self):
        """Kết nối tới Elasticsearch"""
        try:
            es_config = {
                "hosts": [f"http://{settings.ELASTIC_HOST}:{settings.ELASTIC_PORT}"]
            }
            
            # Thêm authentication nếu có
            if settings.ELASTIC_USER and settings.ELASTIC_PASSWORD:
                es_config["basic_auth"] = (
                    settings.ELASTIC_USER,
                    settings.ELASTIC_PASSWORD
                )
            
            self._client = Elasticsearch(**es_config)
            
            # Kiểm tra kết nối
            if self._client.ping():
                logger.info(f"Connected to Elasticsearch at {settings.ELASTIC_HOST}:{settings.ELASTIC_PORT}")
                self._init_index()
            else:
                raise ConnectionError("Cannot ping Elasticsearch")
                
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
    
    def _init_index(self):
        """Khởi tạo index nếu chưa tồn tại"""
        index_name = settings.ELASTIC_INDEX
        
        if not self._client.indices.exists(index=index_name):
            # Định nghĩa mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "anime_id": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "title_english": {"type": "text"},
                        "title_japanese": {"type": "text"},
                        "title_vietnamese": {"type": "text"},
                        "genres": {"type": "keyword"},
                        "year": {"type": "integer"},
                        "episodes": {"type": "integer"},
                        "rating": {"type": "float"},
                        "description": {"type": "text"},
                        "studio": {"type": "keyword"},
                        "studios": {"type": "keyword"},
                        "transcript": {
                            "type": "nested",
                            "properties": {
                                "text": {"type": "text"},
                                "start_time": {"type": "float"},
                                "end_time": {"type": "float"}
                            }
                        },
                        "frames": {
                            "type": "nested",
                            "properties": {
                                "frame_id": {"type": "keyword"},
                                "episode": {"type": "integer"},
                                "timestamp": {"type": "float"},
                                "frame_path": {"type": "keyword"}
                            }
                        }
                    }
                }
            }
            
            self._client.indices.create(index=index_name, body=mapping)
            logger.info(f"Created index: {index_name}")
        else:
            logger.info(f"Index already exists: {index_name}")
    
    def index_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """
        Index một document
        
        Args:
            doc_id: Document ID (anime_id)
            document: Document data
            
        Returns:
            Success status
        """
        if not self._client:
            raise RuntimeError("Elasticsearch client not initialized")
        
        try:
            result = self._client.index(
                index=settings.ELASTIC_INDEX,
                id=doc_id,
                document=document
            )
            logger.info(f"Indexed document: {doc_id}")
            return result["result"] in ["created", "updated"]
        except Exception as e:
            logger.error(f"Failed to index document {doc_id}: {e}")
            raise
    
    def bulk_index(self, documents: List[Dict[str, Any]]) -> tuple:
        """
        Bulk index nhiều documents
        
        Args:
            documents: List of documents, mỗi document phải có '_id' field
            
        Returns:
            (success_count, failed_count)
        """
        if not self._client:
            raise RuntimeError("Elasticsearch client not initialized")
        
        try:
            actions = [
                {
                    "_index": settings.ELASTIC_INDEX,
                    "_id": doc.pop("_id"),
                    "_source": doc
                }
                for doc in documents
            ]
            
            success, failed = bulk(self._client, actions)
            logger.info(f"Bulk indexed: {success} success, {len(failed)} failed")
            return success, len(failed)
        except Exception as e:
            logger.error(f"Bulk index failed: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Lấy document theo ID"""
        if not self._client:
            raise RuntimeError("Elasticsearch client not initialized")
        
        try:
            result = self._client.get(
                index=settings.ELASTIC_INDEX,
                id=doc_id
            )
            return result["_source"]
        except Exception as e:
            logger.warning(f"Document {doc_id} not found: {e}")
            return None
    
    def search(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm documents
        
        Args:
            query: Text query (optional)
            filters: Dictionary of filters (optional)
            size: Số lượng kết quả
            
        Returns:
            List of matching documents
        """
        if not self._client:
            raise RuntimeError("Elasticsearch client not initialized")
        
        try:
            # Build query
            must_clauses = []
            
            if query:
                # Search in text fields (compatible with flat frame schema)
                # Fields: title, description, anime_id
                must_clauses.append({
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "description^2", "anime_id"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                })
            
            if filters:
                for field, value in filters.items():
                    if isinstance(value, list):
                        must_clauses.append({"terms": {field: value}})
                    else:
                        must_clauses.append({"term": {field: value}})
            
            body = {
                "query": {
                    "bool": {
                        "must": must_clauses if must_clauses else [{"match_all": {}}]
                    }
                },
                "size": size
            }
            
            result = self._client.search(
                index=settings.ELASTIC_INDEX,
                body=body
            )
            
            # Return full document including _score for ranking
            docs = []
            for hit in result["hits"]["hits"]:
                doc = hit["_source"]
                doc["_score"] = hit.get("_score", 1.0)
                docs.append(doc)
            return docs
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def update_document(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Update một document"""
        if not self._client:
            raise RuntimeError("Elasticsearch client not initialized")
        
        try:
            result = self._client.update(
                index=settings.ELASTIC_INDEX,
                id=doc_id,
                body={"doc": updates}
            )
            logger.info(f"Updated document: {doc_id}")
            return result["result"] == "updated"
        except Exception as e:
            logger.error(f"Update failed for {doc_id}: {e}")
            raise
    
    def delete_document(self, doc_id: str) -> bool:
        """Xóa document theo ID"""
        if not self._client:
            raise RuntimeError("Elasticsearch client not initialized")
        
        try:
            result = self._client.delete(
                index=settings.ELASTIC_INDEX,
                id=doc_id
            )
            logger.info(f"Deleted document: {doc_id}")
            return result["result"] == "deleted"
        except Exception as e:
            logger.error(f"Delete failed for {doc_id}: {e}")
            raise
    
    def get_stats(self) -> dict:
        """Lấy thống kê index"""
        if not self._client:
            return {"error": "Client not initialized"}
        
        try:
            stats = self._client.indices.stats(index=settings.ELASTIC_INDEX)
            return {
                "index": settings.ELASTIC_INDEX,
                "doc_count": stats["_all"]["primaries"]["docs"]["count"],
                "size_in_bytes": stats["_all"]["primaries"]["store"]["size_in_bytes"]
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Đóng kết nối"""
        if self._client:
            self._client.close()
            logger.info("Disconnected from Elasticsearch")


# Singleton instance
elastic_client = ElasticsearchClient()
