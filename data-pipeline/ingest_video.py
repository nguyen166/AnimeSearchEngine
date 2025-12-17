"""
Video Ingestion Script for Anime Search Engine
Extracts keyframes from videos, generates embeddings via AI Service,
and stores data in Milvus (vectors) and Elasticsearch (metadata).
"""

import os
import sys
import json
import cv2
import base64
import hashlib
import logging
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Scene detection for smart keyframe extraction
from scenedetect import detect, ContentDetector, open_video

# Import centralized core modules
from core import settings, MilvusClientWrapper, ElasticClientWrapper

# Configure logging with UTF-8 support for Windows
def setup_logging():
    """Configure logging with proper UTF-8 encoding for Windows compatibility"""
    log_level = getattr(logging, settings.LOG_LEVEL)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Create handlers with UTF-8 encoding
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(logging.Formatter(log_format))
    
    # Force UTF-8 for file handler
    file_handler = logging.FileHandler(settings.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[stream_handler, file_handler]
    )
    
    # Set stdout encoding for Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

setup_logging()
logger = logging.getLogger(__name__)

# Log configuration on startup
settings.log_settings()


# ============================================================================
# AI Service Client
# ============================================================================

class AIServiceClient:
    """Client for external AI embedding service"""
    
    def __init__(self):
        self.url = settings.AI_SERVICE_URL
        self.model = settings.AI_MODEL
        self.timeout = settings.AI_SERVICE_TIMEOUT
    
    def get_embeddings(self, b64_images: List[str]) -> List[List[float]]:
        """
        Get embeddings from AI service
        
        Args:
            b64_images: List of base64 encoded images
            
        Returns:
            List of embedding vectors
        """
        if not b64_images:
            return []
        
        payload = {
            "model": self.model,
            "b64_images": b64_images
        }
        
        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"AI Service returned {response.status_code}: {response.text}")
            
            result = response.json()
            embeddings = [item["embedding"] for item in result.get("data", [])]
            
            return embeddings
            
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to AI Service at {self.url}")
        except requests.exceptions.Timeout:
            raise Exception(f"AI Service timeout after {self.timeout}s")
        except Exception as e:
            raise Exception(f"AI Service error: {e}")
    
    def health_check(self) -> bool:
        """Check if AI service is available"""
        try:
            # Try health endpoint first
            health_url = self.url.replace("/v1/embeddings", "/health")
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                return True
            
            # Fallback: try root endpoint
            root_url = self.url.rsplit("/v1", 1)[0]
            response = requests.get(root_url, timeout=10)
            if response.status_code == 200:
                return True
            
            # Last resort: just check if port is open
            import socket
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((parsed.hostname, parsed.port))
            sock.close()
            return result == 0
            
        except Exception as e:
            logger.debug(f"Health check error: {e}")
            return False


# ============================================================================
# Video Processor
# ============================================================================

class VideoProcessor:
    """Process videos and extract keyframes"""
    
    def __init__(self, ai_client: AIServiceClient, milvus: MilvusClientWrapper, elastic: ElasticClientWrapper):
        self.ai_client = ai_client
        self.milvus = milvus
        self.elastic = elastic
        self.processed_videos = self._load_processed_log()
    
    def _load_processed_log(self) -> set:
        """Load list of already processed videos"""
        processed = set()
        log_path = Path(settings.PROCESSED_LOG)
        
        if log_path.exists():
            with open(log_path, 'r') as f:
                processed = set(line.strip() for line in f if line.strip())
        
        return processed
    
    def _save_processed(self, video_path: str):
        """Mark video as processed"""
        self.processed_videos.add(video_path)
        with open(settings.PROCESSED_LOG, 'a') as f:
            f.write(f"{video_path}\n")
    
    def _generate_frame_id(self, anime_id: str, episode: int, timestamp: float) -> str:
        """Generate unique frame ID"""
        unique_str = f"{anime_id}_ep{episode:03d}_t{timestamp:.2f}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    def _frame_to_base64(self, frame) -> str:
        """Convert OpenCV frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    def _load_sidecar_metadata(self, json_path: Path, fallback_anime_id: str, fallback_episode: int) -> Dict[str, Any]:
        """
        Load rich metadata from sidecar JSON file
        
        Args:
            json_path: Path to the .json sidecar file
            fallback_anime_id: Default anime_id if JSON doesn't exist
            fallback_episode: Default episode if JSON doesn't exist
            
        Returns:
            Dictionary with metadata fields:
            - anime_id, episode, title, description, source_url, video_url
        """
        default_metadata = {
            "anime_id": fallback_anime_id,
            "episode": fallback_episode,
            "title": json_path.stem,  # Filename as title
            "description": "",
            "source_url": "",
            "video_url": ""
        }
        
        if not json_path.exists():
            logger.info(f"📄 No sidecar JSON found at {json_path}, using filename metadata")
            return default_metadata
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info(f"📄 Loaded metadata from {json_path}")
            
            # Merge with defaults (JSON values take precedence)
            return {
                "anime_id": metadata.get("anime_id", fallback_anime_id),
                "episode": metadata.get("episode", fallback_episode),
                "title": metadata.get("title", default_metadata["title"]),
                "description": metadata.get("description", ""),
                "source_url": metadata.get("source_url", ""),
                "video_url": metadata.get("video_url", "")
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Invalid JSON in {json_path}: {e}")
            return default_metadata
        except Exception as e:
            logger.warning(f"⚠️ Error reading {json_path}: {e}")
            return default_metadata

    def _parse_video_info(self, video_path: str) -> Tuple[str, int, str]:
        """
        Parse video filename to extract anime_id, episode, season
        
        Expected formats:
        - anime_name_ep01.mp4
        - anime_name_s01e01.mp4
        - anime_name_episode_1.mp4
        
        Returns:
            (anime_id, episode, season)
        """
        filename = Path(video_path).stem.lower()
        
        # Try to extract episode number
        import re
        
        # Pattern: _ep01, _e01, _episode01, _episode_01
        ep_match = re.search(r'[_\-](?:ep|e|episode)[_\-]?(\d+)', filename)
        if ep_match:
            episode = int(ep_match.group(1))
            anime_id = filename[:ep_match.start()]
        else:
            # Pattern: _s01e01
            se_match = re.search(r'[_\-]s(\d+)e(\d+)', filename)
            if se_match:
                season = f"S{int(se_match.group(1)):02d}"
                episode = int(se_match.group(2))
                anime_id = filename[:se_match.start()]
                return anime_id.replace('_', '-'), episode, season
            else:
                # Fallback: use filename as anime_id, episode 1
                anime_id = filename
                episode = 1
        
        # Clean anime_id
        anime_id = anime_id.strip('_-').replace('_', '-')
        
        return anime_id, episode, ""
    
    def extract_frames(self, video_path: str, frame_interval: float = 1.0) -> List[Tuple[float, Any]]:
        """
        Extract keyframes from video using Scene Detection (AIC Standard)
        
        Strategy:
        - Use ContentDetector to find scene boundaries
        - Extract the middle frame of each scene (best representation)
        - For long scenes (>5s), also extract frames every 2s (adaptive sampling)
        
        Args:
            video_path: Path to video file
            frame_interval: Fallback interval if scene detection fails (seconds)
            
        Returns:
            List of (timestamp, frame) tuples
        """
        frames = []
        
        # Open video with scenedetect
        video = open_video(video_path)
        fps = video.frame_rate
        total_frames = video.duration.get_frames()
        duration = video.duration.get_seconds()
        
        logger.info(f"📹 Video: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")
        
        # Detect scenes using ContentDetector
        # threshold=27.0: Good for anime style (high contrast scenes)
        # min_scene_len=15: Minimum 15 frames per scene (~0.5s at 30fps)
        logger.info("🎬 Running scene detection (ContentDetector, threshold=27.0)...")
        
        try:
            scene_list = detect(
                video_path,
                ContentDetector(threshold=27.0, min_scene_len=15)
            )
            logger.info(f"🎯 Detected {len(scene_list)} scenes")
        except Exception as e:
            # Handle variable resolution or corrupt videos
            error_msg = str(e).lower()
            if "incorrect size" in error_msg or "corrupt" in error_msg or "resolution" in error_msg:
                logger.warning(f"⚠️ Video has variable resolution or is corrupt: {e}")
                logger.warning("⚠️ Falling back to uniform sampling (more tolerant of video issues).")
            else:
                logger.warning(f"⚠️ Scene detection failed: {e}.")
            return self._extract_frames_uniform(video_path, frame_interval)
        
        # If no scenes detected, fallback to uniform sampling
        if not scene_list:
            logger.warning("⚠️ No scenes detected. Falling back to uniform sampling.")
            return self._extract_frames_uniform(video_path, frame_interval)
        
        # Open video with OpenCV for frame extraction
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        # Extract keyframes from each scene
        for i, (start_time, end_time) in enumerate(scene_list):
            scene_start = start_time.get_seconds()
            scene_end = end_time.get_seconds()
            scene_duration = scene_end - scene_start
            
            # Calculate timestamps to extract
            timestamps_to_extract = []
            
            # 1. Always extract middle frame (best representation)
            middle_timestamp = scene_start + (scene_duration / 2)
            timestamps_to_extract.append(middle_timestamp)
            
            # 2. For long scenes (>5s), add adaptive sampling every 2s
            if scene_duration > 5.0:
                adaptive_interval = 2.0
                current_ts = scene_start + adaptive_interval
                while current_ts < scene_end - 1.0:  # Stop 1s before end
                    # Avoid duplicates near middle frame
                    if abs(current_ts - middle_timestamp) > 1.0:
                        timestamps_to_extract.append(current_ts)
                    current_ts += adaptive_interval
            
            # Sort timestamps
            timestamps_to_extract.sort()
            
            # Extract frames at calculated timestamps
            for timestamp in timestamps_to_extract:
                frame_number = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    frames.append((timestamp, frame))
        
        cap.release()
        logger.info(f"📷 Extracted {len(frames)} keyframes from {len(scene_list)} scenes")
        
        return frames
    
    def _extract_frames_uniform(self, video_path: str, frame_interval: float = 1.0) -> List[Tuple[float, Any]]:
        """
        Fallback: Extract frames at uniform intervals
        More tolerant of variable resolution / corrupt videos
        
        Args:
            video_path: Path to video file
            frame_interval: Extract 1 frame every N seconds
            
        Returns:
            List of (timestamp, frame) tuples
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps * frame_interval)
        frame_count = 0
        skipped_frames = 0
        target_size = None  # Will be set from first valid frame
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                # Validate frame
                if frame is None or frame.size == 0:
                    skipped_frames += 1
                    frame_count += 1
                    continue
                
                # Set target size from first frame, or resize to match
                current_size = (frame.shape[1], frame.shape[0])  # (width, height)
                
                if target_size is None:
                    target_size = current_size
                elif current_size != target_size:
                    # Resize to target size for consistency
                    try:
                        frame = cv2.resize(frame, target_size)
                    except Exception:
                        skipped_frames += 1
                        frame_count += 1
                        continue
                
                timestamp = frame_count / fps
                frames.append((timestamp, frame))
            
            frame_count += 1
        
        cap.release()
        
        if skipped_frames > 0:
            logger.warning(f"⚠️ Skipped {skipped_frames} invalid/mismatched frames")
        
        logger.info(f"📷 Extracted {len(frames)} frames (uniform sampling)")
        
        return frames
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a single video file
        
        Returns:
            Statistics about the processing
        """
        video_path = str(Path(video_path).resolve())
        
        # Skip if already processed
        if video_path in self.processed_videos:
            logger.info(f"⏭️  Skipping already processed: {video_path}")
            return {"status": "skipped", "frames": 0}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎬 Processing: {video_path}")
        logger.info(f"{'='*60}")
        
        # Parse video info from filename (fallback)
        anime_id, episode, season = self._parse_video_info(video_path)
        
        # Load rich metadata from sidecar JSON if available
        json_path = Path(video_path).with_suffix('.json')
        metadata = self._load_sidecar_metadata(json_path, anime_id, episode)
        
        # Override with JSON values if available
        anime_id = metadata.get("anime_id", anime_id)
        episode = metadata.get("episode", episode)
        title = metadata.get("title", Path(video_path).stem)
        description = metadata.get("description", "")
        source_url = metadata.get("source_url", "")
        video_url = metadata.get("video_url", "")
        
        logger.info(f"📺 Anime: {anime_id}, Episode: {episode}, Season: {season}")
        logger.info(f"📝 Title: {title}")
        if source_url:
            logger.info(f"🔗 Source: {source_url}")
        
        # Extract frames
        frames = self.extract_frames(video_path, settings.FRAME_INTERVAL)
        
        if not frames:
            logger.warning(f"⚠️  No frames extracted from {video_path}")
            return {"status": "no_frames", "frames": 0}
        
        # Process in batches
        total_inserted = 0
        batch_size = settings.BATCH_SIZE
        
        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            
            logger.info(f"🔄 Processing batch {batch_start//batch_size + 1}: frames {batch_start+1}-{batch_end}")
            
            # Convert frames to base64
            b64_images = [self._frame_to_base64(frame) for _, frame in batch_frames]
            
            # Get embeddings from AI service
            try:
                embeddings = self.ai_client.get_embeddings(b64_images)
                
                # Debug: log embedding structure
                if embeddings:
                    logger.debug(f"📊 Embeddings received: count={len(embeddings)}, first_dim={len(embeddings[0]) if embeddings[0] else 'N/A'}")
                    
            except Exception as e:
                logger.error(f"❌ AI Service error: {e}")
                raise
            
            if len(embeddings) != len(batch_frames):
                logger.warning(f"⚠️  Embedding count mismatch: {len(embeddings)} vs {len(batch_frames)} frames")
                # Try to recover by taking only matching count
                min_count = min(len(embeddings), len(batch_frames))
                embeddings = embeddings[:min_count]
                batch_frames = batch_frames[:min_count]
                if min_count == 0:
                    continue
            
            # Prepare data for storage
            milvus_data = []
            elastic_data = []
            
            # Create frame output directory
            frame_dir = Path(settings.VIDEO_DIR).parent / "frames" / anime_id / f"ep{episode:03d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            
            for i, ((timestamp, frame), embedding) in enumerate(zip(batch_frames, embeddings)):
                frame_id = self._generate_frame_id(anime_id, episode, timestamp)
                
                # Save frame to disk
                frame_filename = f"frame_{timestamp:.2f}.jpg"
                frame_abs_path = frame_dir / frame_filename
                cv2.imwrite(str(frame_abs_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # Relative path for API serving
                frame_rel_path = f"frames/{anime_id}/ep{episode:03d}/{frame_filename}"
                
                milvus_data.append({
                    "id": frame_id,
                    "anime_id": anime_id,
                    "episode": episode,
                    "timestamp": timestamp,
                    "season": season,
                    "embedding": embedding
                })
                
                elastic_data.append({
                    "id": frame_id,
                    "anime_id": anime_id,
                    "episode": episode,
                    "timestamp": timestamp,
                    "season": season,
                    "file_path": video_path,
                    "frame_path": frame_rel_path,
                    "created_at": datetime.utcnow().isoformat(),
                    # Rich metadata from sidecar JSON
                    "title": title,
                    "description": description,
                    "source_url": source_url,
                    "video_url": video_url
                })
            
            # Insert into databases
            try:
                milvus_count = self.milvus.insert(milvus_data)
                elastic_count = self.elastic.bulk_index(elastic_data)
                total_inserted += milvus_count
                logger.info(f"✅ Batch saved: {len(milvus_data)} frames to {frame_dir}")
                logger.info(f"✅ Inserted: Milvus={milvus_count}, Elasticsearch={elastic_count}")
            except Exception as e:
                logger.error(f"❌ Database insert error: {e}")
                raise
        
        # Mark as processed
        self._save_processed(video_path)
        
        logger.info(f"✅ Completed: {video_path} - {total_inserted} frames inserted")
        
        return {
            "status": "success",
            "frames": total_inserted,
            "anime_id": anime_id,
            "episode": episode
        }
    
    def process_directory(self, directory: str) -> Dict[str, Any]:
        """
        Process all videos in a directory
        
        Returns:
            Statistics about the processing
        """
        video_dir = Path(directory)
        
        if not video_dir.exists():
            logger.error(f"❌ Directory not found: {directory}")
            return {"status": "error", "message": "Directory not found"}
        
        # Find all video files
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f'**/*{ext}'))
        
        logger.info(f"📂 Found {len(video_files)} video files in {directory}")
        
        if not video_files:
            return {"status": "no_videos", "processed": 0}
        
        # Process each video
        results = {
            "total": len(video_files),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "total_frames": 0,
            "details": []
        }
        
        for video_path in video_files:
            try:
                result = self.process_video(str(video_path))
                
                if result["status"] == "success":
                    results["processed"] += 1
                    results["total_frames"] += result["frames"]
                elif result["status"] == "skipped":
                    results["skipped"] += 1
                
                results["details"].append({
                    "file": str(video_path),
                    **result
                })
                
            except Exception as e:
                logger.error(f"❌ Failed to process {video_path}: {e}")
                results["failed"] += 1
                results["details"].append({
                    "file": str(video_path),
                    "status": "error",
                    "error": str(e)
                })
        
        return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Video Ingestion Script for Anime Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_video.py                           # Process default VIDEO_DIR
  python ingest_video.py --dir ./my_videos         # Process specific directory
  python ingest_video.py --video ./video.mp4       # Process single video
  python ingest_video.py --check                   # Health check services
        """
    )
    
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default=settings.VIDEO_DIR,
        help=f'Directory containing video files (default: {settings.VIDEO_DIR})'
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        help='Process a single video file'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check health of all services'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=settings.BATCH_SIZE,
        help=f'Batch size for processing (default: {settings.BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=settings.FRAME_INTERVAL,
        help=f'Frame extraction interval in seconds (default: {settings.FRAME_INTERVAL})'
    )
    
    args = parser.parse_args()
    
    # Update config from args
    # Note: settings is immutable, args will override at runtime
    # Update at call sites instead
    
    logger.info("🚀 Starting Video Ingestion Pipeline")
    logger.info(f"📋 Configuration:")
    logger.info(f"   AI Service: {settings.AI_SERVICE_URL}")
    logger.info(f"   Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    logger.info(f"   Elasticsearch: {settings.ELASTIC_HOST}:{settings.ELASTIC_PORT}")
    logger.info(f"   Batch Size: {args.batch_size}")
    logger.info(f"   Frame Interval: {args.interval}s")
    
    # Initialize clients
    try:
        ai_client = AIServiceClient()
        
        # Health check mode
        if args.check:
            logger.info("\n🔍 Running health checks...")
            
            ai_ok = ai_client.health_check()
            logger.info(f"   AI Service: {'✅ OK' if ai_ok else '❌ DOWN'}")
            
            try:
                milvus = MilvusClientWrapper()
                logger.info(f"   Milvus: ✅ OK ({milvus.count()} entities)")
                milvus.close()
            except Exception as e:
                logger.info(f"   Milvus: ❌ {e}")
            
            try:
                elastic = ElasticClientWrapper()
                logger.info(f"   Elasticsearch: ✅ OK")
                elastic.close()
            except Exception as e:
                logger.info(f"   Elasticsearch: ❌ {e}")
            
            return
        
        # Check AI service before starting
        if not ai_client.health_check():
            logger.error("❌ AI Service is not available. Aborting.")
            sys.exit(1)
        
        milvus = MilvusClientWrapper()
        elastic = ElasticClientWrapper()
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        sys.exit(1)
    
    # Initialize processor
    processor = VideoProcessor(ai_client, milvus, elastic)
    
    try:
        if args.video:
            # Process single video
            result = processor.process_video(args.video)
            logger.info(f"\n📊 Result: {result}")
        else:
            # Process directory
            results = processor.process_directory(args.dir)
            
            logger.info(f"\n{'='*60}")
            logger.info("📊 Processing Summary")
            logger.info(f"{'='*60}")
            logger.info(f"   Total videos: {results['total']}")
            logger.info(f"   Processed: {results['processed']}")
            logger.info(f"   Skipped: {results['skipped']}")
            logger.info(f"   Failed: {results['failed']}")
            logger.info(f"   Total frames: {results['total_frames']}")
            logger.info(f"   Milvus entities: {milvus.count()}")
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        sys.exit(1)
    finally:
        milvus.close()
        elastic.close()
        logger.info("👋 Done!")


if __name__ == "__main__":
    main()
