"""
Milvus Vector Database Connection
Quản lý kết nối và thao tác với Milvus vector database
"""

from typing import List, Optional
import logging
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from app.config import settings

logger = logging.getLogger(__name__)


class MilvusClient:
    """Singleton class để quản lý kết nối Milvus"""
    
    _instance = None
    _collection: Optional[Collection] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MilvusClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._connect()
    
    def _connect(self):
        """Kết nối tới Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT
            )
            logger.info(f"Connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
            self._init_collection()
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _init_collection(self):
        """Khởi tạo hoặc load collection"""
        collection_name = settings.MILVUS_COLLECTION
        
        # Kiểm tra collection có tồn tại không
        if utility.has_collection(collection_name):
            self._collection = Collection(collection_name)
            self._collection.load()
            logger.info(f"Loaded existing collection: {collection_name}")
        else:
            logger.warning(f"Collection {collection_name} does not exist. Creating new one...")
            self._create_collection(collection_name)
    
    def _create_collection(self, collection_name: str):
        """Tạo collection mới với schema"""
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="anime_id", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="episode", dtype=DataType.INT32),
            FieldSchema(name="timestamp", dtype=DataType.FLOAT),
            FieldSchema(name="season", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.VECTOR_DIM)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Anime frame embeddings"
        )
        
        # Tạo collection
        self._collection = Collection(
            name=collection_name,
            schema=schema
        )
        
        # Tạo index cho vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        self._collection.load()
        logger.info(f"Created and indexed collection: {collection_name}")
    
    def insert(self, data: List[dict]) -> List[str]:
        """
        Insert dữ liệu vào collection
        
        Args:
            data: List of dictionaries với keys: id, anime_id, episode, timestamp, season, embedding
            
        Returns:
            List of inserted IDs
        """
        if not self._collection:
            raise RuntimeError("Collection not initialized")
        
        try:
            entities = [
                [item["id"] for item in data],
                [item["anime_id"] for item in data],
                [item["episode"] for item in data],
                [item["timestamp"] for item in data],
                [item.get("season", "") for item in data],
                [item["embedding"] for item in data]
            ]
            
            insert_result = self._collection.insert(entities)
            self._collection.flush()
            logger.info(f"Inserted {len(data)} vectors into Milvus")
            return insert_result.primary_keys
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            raise
    
    def search(
        self,
        query_vectors: List[List[float]],
        top_k: int = 10,
        filters: Optional[str] = None
    ) -> List[List[dict]]:
        """
        Tìm kiếm vectors tương tự
        
        Args:
            query_vectors: List of query vectors
            top_k: Số lượng kết quả trả về
            filters: Filter expression (optional)
            
        Returns:
            List of search results
        """
        if not self._collection:
            raise RuntimeError("Collection not initialized")
        
        try:
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            results = self._collection.search(
                data=query_vectors,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filters,
                output_fields=["anime_id", "episode", "timestamp", "season"]
            )
            
            # Format kết quả
            formatted_results = []
            for hits in results:
                hit_list = []
                for hit in hits:
                    hit_list.append({
                        "id": hit.id,
                        "score": hit.distance,
                        "anime_id": hit.entity.get("anime_id"),
                        "episode": hit.entity.get("episode"),
                        "timestamp": hit.entity.get("timestamp"),
                        "season": hit.entity.get("season")
                    })
                formatted_results.append(hit_list)
            
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def delete(self, ids: List[str]):
        """Xóa vectors theo IDs"""
        if not self._collection:
            raise RuntimeError("Collection not initialized")
        
        try:
            expr = f"id in {ids}"
            self._collection.delete(expr)
            logger.info(f"Deleted {len(ids)} vectors")
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise
    
    def get_stats(self) -> dict:
        """Lấy thống kê collection"""
        if not self._collection:
            return {"error": "Collection not initialized"}
        
        return {
            "name": self._collection.name,
            "num_entities": self._collection.num_entities,
            "loaded": utility.load_state(self._collection.name)
        }
    
    def close(self):
        """Đóng kết nối"""
        connections.disconnect("default")
        logger.info("Disconnected from Milvus")


# Singleton instance
milvus_client = MilvusClient()
