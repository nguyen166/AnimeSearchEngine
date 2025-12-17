"""
Milvus Vector Database Client Wrapper
Centralized Milvus connection and operations
"""

import logging
from typing import Dict, Any, List, Optional
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)

from .config import settings

logger = logging.getLogger(__name__)


class MilvusClientWrapper:
    """
    Milvus vector database client wrapper
    
    Handles:
    - Connection management
    - Collection initialization and schema
    - Data insertion and flushing
    - Collection statistics
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        alias: str = "default"
    ):
        """
        Initialize Milvus client
        
        Args:
            collection_name: Name of the collection (defaults to settings.MILVUS_COLLECTION)
            alias: Connection alias
        """
        self.collection_name = collection_name or settings.MILVUS_COLLECTION
        self.alias = alias
        self._collection: Optional[Collection] = None
        
        self._connect()
        self._init_collection()
    
    def _connect(self):
        """Connect to Milvus server"""
        try:
            conn_params = {
                "alias": self.alias,
                "host": settings.MILVUS_HOST,
                "port": settings.MILVUS_PORT
            }
            
            # Add authentication if configured
            if settings.MILVUS_USER:
                conn_params["user"] = settings.MILVUS_USER
            if settings.MILVUS_PASSWORD:
                conn_params["password"] = settings.MILVUS_PASSWORD
            
            connections.connect(**conn_params)
            logger.info(f"✅ Connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Milvus: {e}")
            raise
    
    def _init_collection(self):
        """Initialize or load collection"""
        if utility.has_collection(self.collection_name):
            self._collection = Collection(self.collection_name)
            self._collection.load()
            logger.info(f"📦 Loaded existing collection: {self.collection_name}")
        else:
            logger.info(f"📦 Creating new collection: {self.collection_name}")
            self._create_collection()
    
    def _create_collection(self):
        """Create new collection with schema"""
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=100
            ),
            FieldSchema(
                name="anime_id",
                dtype=DataType.VARCHAR,
                max_length=100
            ),
            FieldSchema(
                name="episode",
                dtype=DataType.INT32
            ),
            FieldSchema(
                name="timestamp",
                dtype=DataType.FLOAT
            ),
            FieldSchema(
                name="season",
                dtype=DataType.VARCHAR,
                max_length=50
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=settings.VECTOR_DIM
            )
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Anime frame embeddings"
        )
        
        self._collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        # Create index for vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        self._collection.load()
        logger.info(f"✅ Created and indexed collection: {self.collection_name}")
    
    def insert(self, data: List[Dict[str, Any]]) -> int:
        """
        Insert data into collection
        
        Args:
            data: List of dictionaries with keys:
                - id: Unique frame ID
                - anime_id: Anime identifier
                - episode: Episode number
                - timestamp: Frame timestamp in seconds
                - season: Season identifier (optional)
                - embedding: Vector embedding
        
        Returns:
            Number of inserted entities
        """
        if not self._collection or not data:
            return 0
        
        try:
            # Validate and flatten embeddings if needed
            embeddings = []
            for item in data:
                emb = item["embedding"]
                # Handle nested list (e.g., [[1,2,3,...]])
                if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                    emb = emb[0]
                # Validate dimension
                if len(emb) != settings.VECTOR_DIM:
                    logger.warning(f"⚠️ Embedding dimension mismatch: got {len(emb)}, expected {settings.VECTOR_DIM}")
                embeddings.append(emb)
            
            entities = [
                [item["id"] for item in data],
                [item["anime_id"] for item in data],
                [item["episode"] for item in data],
                [item["timestamp"] for item in data],
                [item.get("season", "") for item in data],
                embeddings
            ]
            
            self._collection.insert(entities)
            self._collection.flush()
            
            logger.debug(f"✅ Inserted {len(data)} entities into {self.collection_name}")
            return len(data)
            
        except Exception as e:
            logger.error(f"❌ Milvus insert failed: {e}")
            raise
    
    def count(self) -> int:
        """
        Get total entity count in collection
        
        Returns:
            Number of entities
        """
        if self._collection:
            return self._collection.num_entities
        return 0
    
    def drop_collection(self):
        """Drop the collection (use with caution!)"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.warning(f"🗑️ Dropped collection: {self.collection_name}")
    
    def close(self):
        """Close Milvus connection"""
        try:
            connections.disconnect(self.alias)
            logger.info("👋 Disconnected from Milvus")
        except Exception as e:
            logger.warning(f"Error disconnecting from Milvus: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
