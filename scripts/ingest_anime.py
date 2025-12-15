"""
Ingest Anime Script
Script để cắt ảnh từ video và nạp dữ liệu vào database
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import cv2
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import settings
from app.core.milvus import milvus_client
from app.core.elastic import elastic_client
from app.services.embedding import embedding_service
from app.models.schemas import AnimeMetadata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnimeIngestor:
    """Class để xử lý ingestion của anime video"""
    
    def __init__(
        self,
        video_path: str,
        anime_id: str,
        episode: int,
        fps: float = 1.0,
        season: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        self.video_path = video_path
        self.anime_id = anime_id
        self.episode = episode
        self.fps = fps
        self.season = season or ""
        self.output_dir = output_dir or settings.FRAME_DIR
        
        # Create output directory
        self.frame_output_dir = os.path.join(
            self.output_dir,
            anime_id,
            f"ep{episode:02d}"
        )
        os.makedirs(self.frame_output_dir, exist_ok=True)
    
    def extract_frames(self) -> List[str]:
        """
        Extract frames từ video
        
        Returns:
            List of frame paths
        """
        logger.info(f"Extracting frames from {self.video_path}")
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        logger.info(f"Video FPS: {video_fps}, Duration: {duration:.2f}s")
        logger.info(f"Extracting {self.fps} frames per second")
        
        frame_paths = []
        frame_interval = int(video_fps / self.fps)
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at intervals
            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps
                frame_id = f"{self.anime_id}_ep{self.episode:02d}_{saved_count:05d}"
                frame_path = os.path.join(
                    self.frame_output_dir,
                    f"{frame_id}.jpg"
                )
                
                # Save frame
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                saved_count += 1
                
                if saved_count % 100 == 0:
                    logger.info(f"Extracted {saved_count} frames...")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {saved_count} frames total")
        
        return frame_paths
    
    def generate_embeddings(self, frame_paths: List[str]) -> List[dict]:
        """
        Generate embeddings cho các frames
        
        Args:
            frame_paths: List of frame paths
            
        Returns:
            List of data entries for Milvus
        """
        logger.info(f"Generating embeddings for {len(frame_paths)} frames...")
        
        data = []
        batch_size = 32
        
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i+batch_size]
            
            # Load images
            images = []
            for path in batch_paths:
                img = Image.open(path)
                images.append(img)
            
            # Generate embeddings
            embeddings = embedding_service.encode_batch_images(images)
            
            # Prepare data
            for j, (path, embedding) in enumerate(zip(batch_paths, embeddings)):
                frame_idx = i + j
                timestamp = frame_idx / self.fps
                frame_id = Path(path).stem
                
                data.append({
                    "id": frame_id,
                    "anime_id": self.anime_id,
                    "episode": self.episode,
                    "timestamp": timestamp,
                    "season": self.season,
                    "embedding": embedding
                })
            
            if (i + batch_size) % 100 == 0:
                logger.info(f"Processed {min(i + batch_size, len(frame_paths))}/{len(frame_paths)} frames")
        
        logger.info(f"Generated {len(data)} embeddings")
        return data
    
    def index_to_milvus(self, data: List[dict]):
        """Index embeddings vào Milvus"""
        logger.info(f"Indexing {len(data)} vectors to Milvus...")
        
        batch_size = 1000
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            milvus_client.insert(batch)
            logger.info(f"Indexed {min(i+batch_size, len(data))}/{len(data)} vectors")
        
        logger.info("Indexing to Milvus completed")
    
    def index_metadata(self, metadata: AnimeMetadata, frame_paths: List[str]):
        """Index metadata vào Elasticsearch"""
        logger.info(f"Indexing metadata for {self.anime_id}")
        
        # Prepare frames info
        frames_info = []
        for i, path in enumerate(frame_paths):
            frame_id = Path(path).stem
            timestamp = i / self.fps
            frames_info.append({
                "frame_id": frame_id,
                "episode": self.episode,
                "timestamp": timestamp,
                "frame_path": path
            })
        
        # Prepare document
        doc = metadata.dict()
        
        # Check if anime already exists
        existing = elastic_client.get_document(self.anime_id)
        if existing:
            # Update with new frames
            existing_frames = existing.get("frames", [])
            # Remove frames from same episode
            existing_frames = [
                f for f in existing_frames 
                if f.get("episode") != self.episode
            ]
            # Add new frames
            all_frames = existing_frames + frames_info
            elastic_client.update_document(
                self.anime_id,
                {"frames": all_frames}
            )
        else:
            # Create new document
            doc["frames"] = frames_info
            elastic_client.index_document(self.anime_id, doc)
        
        logger.info("Metadata indexing completed")
    
    def run(self, metadata: Optional[AnimeMetadata] = None):
        """
        Chạy toàn bộ pipeline
        
        Args:
            metadata: Anime metadata (optional)
        """
        try:
            # 1. Extract frames
            frame_paths = self.extract_frames()
            
            if not frame_paths:
                logger.warning("No frames extracted!")
                return
            
            # 2. Generate embeddings
            data = self.generate_embeddings(frame_paths)
            
            # 3. Index to Milvus
            self.index_to_milvus(data)
            
            # 4. Index metadata to Elasticsearch
            if metadata:
                self.index_metadata(metadata, frame_paths)
            
            logger.info("✓ Ingestion completed successfully!")
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Ingest anime video into search engine"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--anime-id",
        required=True,
        help="Anime ID"
    )
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode number"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0)"
    )
    parser.add_argument(
        "--title",
        help="Anime title"
    )
    parser.add_argument(
        "--genres",
        nargs="+",
        help="Anime genres"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Release year"
    )
    parser.add_argument(
        "--season",
        help="Anime season (e.g., '2023-Spring')"
    )
    
    args = parser.parse_args()
    
    # Create metadata
    metadata = None
    if args.title:
        metadata = AnimeMetadata(
            anime_id=args.anime_id,
            title=args.title,
            genres=args.genres or [],
            year=args.year,
            episodes=1  # Will be updated
        )
    
    # Create ingestor and run
    ingestor = AnimeIngestor(
        video_path=args.video,
        anime_id=args.anime_id,
        episode=args.episode,
        fps=args.fps,
        season=args.season
    )
    
    ingestor.run(metadata)


if __name__ == "__main__":
    main()
