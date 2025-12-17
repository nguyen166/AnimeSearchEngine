"""
Elasticsearch Client Wrapper
Centralized Elasticsearch connection and operations
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from .config import settings

logger = logging.getLogger(__name__)


class ElasticClientWrapper:
    """
    Elasticsearch metadata client wrapper
    
    Handles:
    - Connection management
    - Index initialization and mapping
    - Bulk document indexing
    - Index statistics
    """
    
    def __init__(
        self,
        index_name: Optional[str] = None
    ):
        """
        Initialize Elasticsearch client
        
        Args:
            index_name: Name of the index (defaults to settings.ELASTIC_INDEX)
        """
        self.index_name = index_name or settings.ELASTIC_INDEX
        self._client: Optional[Elasticsearch] = None
        
        self._connect()
        self._init_index()
    
    def _connect(self):
        """Connect to Elasticsearch"""
        try:
            es_config = {
                "hosts": [settings.elastic_uri]
            }
            
            # Add authentication if configured
            if settings.ELASTIC_USER and settings.ELASTIC_PASSWORD:
                es_config["basic_auth"] = (settings.ELASTIC_USER, settings.ELASTIC_PASSWORD)
            
            self._client = Elasticsearch(**es_config)
            
            # Use info() instead of ping() for better compatibility with ES 8.x
            info = self._client.info()
            if info and "cluster_name" in info:
                logger.info(f"✅ Connected to Elasticsearch at {settings.elastic_uri}")
                logger.info(f"   Cluster: {info['cluster_name']}, Version: {info['version']['number']}")
            else:
                raise ConnectionError("Elasticsearch connection failed")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to Elasticsearch: {e}")
            raise
    
    def _init_index(self):
        """Initialize index with mapping"""
        if not self._client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "anime_id": {"type": "keyword"},
                        "episode": {"type": "integer"},
                        "timestamp": {"type": "float"},
                        "season": {"type": "keyword"},
                        "file_path": {"type": "text"},
                        "frame_path": {"type": "text"},
                        "created_at": {"type": "date"},
                        # Rich metadata from crawler sidecar JSON
                        "title": {"type": "text"},
                        "description": {"type": "text"},
                        "source_url": {"type": "keyword"},
                        "video_url": {"type": "keyword"}
                    }
                }
            }
            
            self._client.indices.create(index=self.index_name, body=mapping)
            logger.info(f"📦 Created Elasticsearch index: {self.index_name}")
        else:
            logger.info(f"📦 Using existing Elasticsearch index: {self.index_name}")
    
    def bulk_index(self, documents: List[Dict[str, Any]]) -> int:
        """
        Bulk index documents
        
        Args:
            documents: List of dictionaries with keys:
                - id: Unique frame ID
                - anime_id: Anime identifier
                - episode: Episode number
                - timestamp: Frame timestamp in seconds
                - season: Season identifier (optional)
                - file_path: Video file path
                - frame_path: Extracted frame path
                - created_at: ISO timestamp (optional, defaults to now)
        
        Returns:
            Number of successfully indexed documents
        """
        if not self._client or not documents:
            return 0
        
        try:
            actions = [
                {
                    "_index": self.index_name,
                    "_id": doc["id"],
                    "_source": {
                        **doc,
                        "created_at": doc.get("created_at", datetime.utcnow().isoformat())
                    }
                }
                for doc in documents
            ]
            
            success, failed = bulk(self._client, actions, raise_on_error=False)
            
            if failed:
                logger.warning(f"⚠️ {len(failed)} documents failed to index")
            
            logger.debug(f"✅ Indexed {success} documents into {self.index_name}")
            return success
            
        except Exception as e:
            logger.error(f"❌ Elasticsearch bulk index failed: {e}")
            raise
    
    def count(self) -> int:
        """
        Get total document count in index
        
        Returns:
            Number of documents
        """
        if self._client:
            try:
                result = self._client.count(index=self.index_name)
                return result["count"]
            except Exception as e:
                logger.error(f"Error counting documents: {e}")
                return 0
        return 0
    
    def delete_index(self):
        """Delete the index (use with caution!)"""
        if self._client and self._client.indices.exists(index=self.index_name):
            self._client.indices.delete(index=self.index_name)
            logger.warning(f"🗑️ Deleted index: {self.index_name}")
    
    def search(
        self,
        query: Dict[str, Any],
        size: int = 10,
        from_: int = 0
    ) -> Dict[str, Any]:
        """
        Search documents
        
        Args:
            query: Elasticsearch query DSL
            size: Number of results to return
            from_: Offset for pagination
        
        Returns:
            Search results
        """
        if not self._client:
            return {"hits": {"hits": []}}
        
        try:
            return self._client.search(
                index=self.index_name,
                body=query,
                size=size,
                from_=from_
            )
        except Exception as e:
            logger.error(f"❌ Elasticsearch search failed: {e}")
            return {"hits": {"hits": []}}
    
    def close(self):
        """Close Elasticsearch connection"""
        try:
            if self._client:
                self._client.close()
                logger.info("👋 Disconnected from Elasticsearch")
        except Exception as e:
            logger.warning(f"Error disconnecting from Elasticsearch: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
