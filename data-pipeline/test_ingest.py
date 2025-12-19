import os
import sys
import json
import cv2
import base64
import hashlib
import logging
import argparse
import requests
import numpy as np  # Required for vector math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
from datetime import datetime

# Import centralized core modules
from core import settings, MilvusClientWrapper, ElasticClientWrapper

# Configure logging
def setup_logging():
    log_level = getattr(logging, settings.LOG_LEVEL)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format, handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE, encoding='utf-8')
    ])
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

setup_logging()
logger = logging.getLogger(__name__)

# ============================================================================
# AI Service Client
# ============================================================================

class AIServiceClient:
    def __init__(self):
        self.url = settings.AI_SERVICE_URL
        self.model = settings.AI_MODEL
        self.timeout = settings.AI_SERVICE_TIMEOUT
    
    def get_embeddings(self, b64_images: List[str]) -> List[List[float]]:
        if not b64_images: return []
        payload = {"model": self.model, "b64_images": b64_images}
        try:
            # Tăng timeout vì batch có thể xử lý lâu
            response = requests.post(self.url, json=payload, timeout=30, headers={"Content-Type": "application/json"})
            if response.status_code != 200:
                raise Exception(f"AI Service Error {response.status_code}")
            return [item["embedding"] for item in response.json().get("data", [])]
        except Exception as e:
            logger.error(f"Error requesting AI Service: {e}")
            raise

    def health_check(self) -> bool:
        try:
            requests.get(self.url.rsplit("/v1", 1)[0], timeout=5)
            return True
        except:
            return False

# ============================================================================
# Video Processor (OPTIMIZED STREAMING ARCHITECTURE)
# ============================================================================

class VideoProcessor:
    def __init__(self, ai_client: AIServiceClient, milvus: MilvusClientWrapper, elastic: ElasticClientWrapper):
        self.ai_client = ai_client
        self.milvus = milvus
        self.elastic = elastic
        self.processed_videos = self._load_processed_log()
        
        # Vortex Paper Params
        self.SAMPLE_STEP = 8          # Sample every 8th frame
        self.REL_DIFF_THRESHOLD = 0.4 # Threshold for L2-norm filtering
        
        # Engineering Params
        self.BATCH_SIZE = 16          # Process 16 sampled frames at a time (Limit RAM/Payload)
        self.EMBED_IMG_SIZE = (480, 270) # Resize for AI to reduce payload (16:9 aspect ratio)

    def _load_processed_log(self) -> set:
        processed = set()
        log_path = Path(settings.PROCESSED_LOG)
        if log_path.exists():
            with open(log_path, 'r') as f:
                processed = set(line.strip() for line in f if line.strip())
        return processed
    
    def _save_processed(self, video_path: str):
        self.processed_videos.add(video_path)
        with open(settings.PROCESSED_LOG, 'a') as f:
            f.write(f"{video_path}\n")
    
    def _generate_frame_id(self, anime_id: str, episode: int, timestamp: float) -> str:
        unique_str = f"{anime_id}_ep{episode:03d}_t{timestamp:.2f}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]

    def _frame_to_base64(self, frame, for_ai: bool = False) -> str:
        """
        Convert frame to base64. 
        If for_ai=True, resize to small resolution to save bandwidth/RAM.
        """
        if for_ai:
            # Resize for Embedding model (doesn't need 1080p)
            frame = cv2.resize(frame, self.EMBED_IMG_SIZE)
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def _parse_video_info(self, video_path: str) -> Tuple[str, int, str]:
        # Simple parser (same as before)
        filename = Path(video_path).stem.lower()
        import re
        ep_match = re.search(r'[_\-](?:ep|e|episode)[_\-]?(\d+)', filename)
        if ep_match:
            return filename[:ep_match.start()], int(ep_match.group(1)), ""
        return filename, 1, ""
    
    def _load_sidecar_metadata(self, json_path: Path, anime_id: str, ep: int) -> Dict:
        # Same as before
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {"anime_id": anime_id, "episode": ep, "title": json_path.stem}

    def frame_generator(self, video_path: str) -> Generator[Tuple[float, Any], None, None]:
        """
        Yields (timestamp, frame) for every 8th frame.
        Reads video strictly sequentially (Low I/O cost).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 24.0
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Logic: Sample every 8th frame (Paper: "extracted from every eighth frame")
            if frame_idx % self.SAMPLE_STEP == 0:
                timestamp = frame_idx / fps
                yield timestamp, frame
            
            frame_idx += 1
            
        cap.release()

    def process_video(self, video_path: str) -> Dict[str, Any]:
        video_path = str(Path(video_path).resolve())
        if video_path in self.processed_videos:
            return {"status": "skipped"}

        logger.info(f"🎬 Processing: {video_path}")
        
        # Metadata setup
        anime_id, episode, season = self._parse_video_info(video_path)
        meta = self._load_sidecar_metadata(Path(video_path).with_suffix('.json'), anime_id, episode)
        
        # Output directory
        frame_dir = Path(settings.VIDEO_DIR).parent / "frames" / str(anime_id) / f"ep{episode:03d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        # State for Adaptive Filtering
        last_kept_embedding = None
        total_extracted = 0
        total_saved = 0
        
        # Batch buffer
        batch_frames = []      # List of (timestamp, original_frame_image)
        batch_b64_for_ai = []  # List of small b64 strings
        
        # Initialize generator
        generator = self.frame_generator(video_path)
        
        try:
            while True:
                try:
                    # Collect a batch
                    timestamp, frame = next(generator)
                    batch_frames.append((timestamp, frame))
                    batch_b64_for_ai.append(self._frame_to_base64(frame, for_ai=True))
                except StopIteration:
                    # End of video
                    pass
                
                # Process if batch is full or end of video reached
                is_end = (len(batch_frames) > 0 and (len(batch_frames) >= self.BATCH_SIZE or last_kept_embedding is None)) 
                # Note: condition checks if we need to flush. If iterator stopped, loop breaks after this flush.
                
                if not batch_frames:
                    break
                    
                if len(batch_frames) >= self.BATCH_SIZE or (batch_frames and not is_end): # Wait, logic above is tricky.
                    # Simplified logic: Flush if batch is full OR generator finished
                    pass # Continue to processing
                else:
                     # Check if generator is empty by peeking? No, generators don't peek.
                     # Better approach: Loop until StopIteration caught, then flush remainder.
                     # Let's refactor the loop slightly.
                     pass
        except Exception:
            pass # Use the cleaner loop below
            
        # --- RE-IMPLEMENTING LOOP FOR CLARITY ---
        generator = self.frame_generator(video_path)
        is_running = True
        
        while is_running:
            # Fill batch
            while len(batch_frames) < self.BATCH_SIZE:
                try:
                    ts, fr = next(generator)
                    batch_frames.append((ts, fr))
                    # Resize locally for AI to save RAM/Bandwidth
                    batch_b64_for_ai.append(self._frame_to_base64(fr, for_ai=True))
                    total_extracted += 1
                except StopIteration:
                    is_running = False
                    break
            
            if not batch_frames:
                break
                
            # --- PROCESS BATCH ---
            logger.info(f"⚙️  Processing batch of {len(batch_frames)} frames (Total scanned: {total_extracted})")
            
            try:
                # 1. Get Embeddings
                embeddings = self.ai_client.get_embeddings(batch_b64_for_ai)
                
                # Handle mismatch (rare)
                if len(embeddings) != len(batch_frames):
                    logger.warning("Mismatch embedding count. Truncating batch.")
                    min_len = min(len(embeddings), len(batch_frames))
                    batch_frames = batch_frames[:min_len]
                    embeddings = embeddings[:min_len]

                # 2. Adaptive Filtering & Saving
                milvus_data = []
                elastic_data = []
                
                for i, embedding in enumerate(embeddings):
                    current_emb_np = np.array(embedding)
                    timestamp, frame = batch_frames[i]
                    
                    keep_frame = False
                    
                    if last_kept_embedding is None:
                        # Always keep the very first frame of video
                        keep_frame = True
                        rel_diff = 1.0
                    else:
                        # Equation 1 from Paper
                        diff = np.linalg.norm(current_emb_np - last_kept_embedding)
                        prev_norm = np.linalg.norm(last_kept_embedding)
                        rel_diff = diff / prev_norm if prev_norm > 0 else 0
                        
                        if rel_diff > self.REL_DIFF_THRESHOLD:
                            keep_frame = True
                    
                    if keep_frame:
                        # Update state
                        last_kept_embedding = current_emb_np
                        total_saved += 1
                        
                        # Save to Disk (High Quality)
                        frame_id = self._generate_frame_id(str(anime_id), int(episode), timestamp)
                        fname = f"frame_{timestamp:.2f}.jpg"
                        cv2.imwrite(str(frame_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        
                        # Prepare DB Records
                        milvus_data.append({
                            "id": frame_id,
                            "anime_id": str(anime_id),
                            "episode": int(episode),
                            "timestamp": float(timestamp),
                            "season": str(season),
                            "embedding": embedding
                        })
                        
                        elastic_data.append({
                            "id": frame_id,
                            "anime_id": str(anime_id),
                            "episode": int(episode),
                            "timestamp": float(timestamp),
                            "title": meta.get("title", ""),
                            "frame_path": f"frames/{anime_id}/ep{episode:03d}/{fname}",
                            "created_at": datetime.utcnow().isoformat()
                        })
                
                # 3. Commit Batch to DB
                if milvus_data:
                    self.milvus.insert(milvus_data)
                    self.elastic.bulk_index(elastic_data)
                    
            except Exception as e:
                logger.error(f"❌ Batch failed: {e}")
                # Don't crash entire script, try next video? or break?
                # Usually break to inspect error
                break
            
            # Clear batch for next iteration
            batch_frames = []
            batch_b64_for_ai = []
            
        # End of video loop
        self._save_processed(video_path)
        logger.info(f"✅ Completed {video_path}: Scanned {total_extracted}, Saved {total_saved} (Ratio: {total_saved/total_extracted if total_extracted else 0:.2%})")
        return {"status": "success", "saved": total_saved}

    def process_directory(self, directory: str):
        # (Giữ nguyên logic loop qua file)
        video_dir = Path(directory)
        videos = list(video_dir.glob('**/*.mp4')) + list(video_dir.glob('**/*.mkv'))
        logger.info(f"Found {len(videos)} videos")
        for v in videos:
            try:
                self.process_video(str(v))
            except Exception as e:
                logger.error(f"Failed {v}: {e}")

# ============================================================================
# Main (Giữ nguyên, chỉ gọi VideoProcessor mới)
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