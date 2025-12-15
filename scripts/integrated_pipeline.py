"""
Integrated Crawl & Ingest Pipeline
Pipeline tích hợp crawl anime và ingest vào database với auto cleanup
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.ingest_anime import AnimeIngestor
from scripts.crawler import crawl_episode
from app.models.schemas import AnimeMetadata
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CrawlConfig:
    """Configuration for crawling"""
    batch_size: int = 24  # Số tập crawl trước khi cleanup
    auto_cleanup: bool = True  # Tự động xóa video sau khi ingest
    keep_frames: bool = True  # Giữ lại frames
    retry_count: int = 3  # Số lần retry khi crawl/ingest fail
    delay_between_episodes: int = 2  # Delay giữa các tập (seconds)


class IntegratedPipeline:
    """Pipeline tích hợp crawl và ingest với auto cleanup"""
    
    def __init__(self, config_path: str, crawl_config: Optional[CrawlConfig] = None):
        """
        Initialize integrated pipeline
        
        Args:
            config_path: Path to JSON config file
            crawl_config: Crawl configuration
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.crawl_config = crawl_config or CrawlConfig()
        
        # Statistics
        self.stats = {
            "total_anime": 0,
            "total_episodes": 0,
            "crawled": 0,
            "ingested": 0,
            "failed": 0,
            "cleaned_up": 0,
            "disk_space_freed": 0
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration từ JSON file"""
        logger.info(f"Loading config from {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config
    
    def run(self):
        """Run integrated pipeline"""
        logger.info("=" * 80)
        logger.info("🚀 Starting Integrated Crawl & Ingest Pipeline")
        logger.info("=" * 80)
        
        self.stats["total_anime"] = len(self.config.get("anime", []))
        
        for anime_config in self.config["anime"]:
            try:
                self._process_anime(anime_config)
            except Exception as e:
                logger.error(f"Failed to process anime: {e}", exc_info=True)
                self.stats["failed"] += 1
        
        self._print_summary()
    
    def _process_anime(self, anime_config: Dict[str, Any]):
        """
        Process một anime: crawl -> ingest -> cleanup
        
        Args:
            anime_config: Anime configuration
        """
        anime_id = anime_config["anime_id"]
        title = anime_config["title"]
        
        logger.info("\n" + "=" * 80)
        logger.info(f"📺 Processing Anime: {title} ({anime_id})")
        logger.info("=" * 80)
        
        # Create metadata
        metadata = AnimeMetadata(
            anime_id=anime_id,
            title=title,
            title_english=anime_config.get("title_english"),
            title_japanese=anime_config.get("title_japanese"),
            title_vietnamese=anime_config.get("title_vietnamese"),
            genres=anime_config.get("genres", []),
            year=anime_config.get("year"),
            episodes=len(anime_config.get("episodes", [])),
            rating=anime_config.get("rating"),
            description=anime_config.get("description"),
            studio=anime_config.get("studio"),
            studios=anime_config.get("studios"),
            season=anime_config.get("season")
        )
        
        episodes = anime_config.get("episodes", [])
        self.stats["total_episodes"] += len(episodes)
        
        logger.info(f"📊 Total episodes: {len(episodes)}")
        logger.info(f"📦 Batch size: {self.crawl_config.batch_size}")
        
        # Process episodes in batches
        batch_number = 0
        for i in range(0, len(episodes), self.crawl_config.batch_size):
            batch = episodes[i:i + self.crawl_config.batch_size]
            batch_number += 1
            
            logger.info(f"\n📦 Processing Batch {batch_number} ({len(batch)} episodes)")
            
            self._process_batch(
                anime_id=anime_id,
                anime_title=title,
                batch=batch,
                metadata=metadata if i == 0 else None,  # Only pass metadata for first batch
                batch_start_idx=i
            )
        
        logger.info(f"✅ Completed all episodes of {title}")
    
    def _process_batch(
        self,
        anime_id: str,
        anime_title: str,
        batch: List[Dict[str, Any]],
        metadata: Optional[AnimeMetadata],
        batch_start_idx: int
    ):
        """
        Process một batch episodes: crawl -> ingest -> cleanup
        
        Args:
            anime_id: Anime ID
            anime_title: Anime title
            batch: Batch of episodes
            metadata: Anime metadata (only for first batch)
            batch_start_idx: Starting index in full episode list
        """
        crawled_videos = []
        
        # Phase 1: Crawl batch
        logger.info(f"\n🌐 Phase 1/3: Crawling {len(batch)} episodes...")
        
        for idx, episode_config in enumerate(batch):
            episode_num = episode_config.get("episode", batch_start_idx + idx + 1)
            
            # Check if video_path exists (already downloaded)
            if "video_path" in episode_config and os.path.exists(episode_config["video_path"]):
                logger.info(f"📹 Episode {episode_num}: Using existing video")
                crawled_videos.append({
                    "episode": episode_num,
                    "video_path": episode_config["video_path"],
                    "fps": episode_config.get("fps", 1.0)
                })
                continue
            
            # Crawl episode
            crawl_url = episode_config.get("crawl_url")
            if not crawl_url:
                logger.warning(f"⚠️  Episode {episode_num}: No crawl_url provided, skipping")
                continue
            
            logger.info(f"🔄 Crawling Episode {episode_num}...")
            
            success = False
            video_path = None
            
            for attempt in range(self.crawl_config.retry_count):
                try:
                    video_path = crawl_episode(
                        url=crawl_url,
                        anime_id=anime_id,
                        episode=f"{episode_num:02d}"
                    )
                    
                    if video_path and os.path.exists(video_path):
                        success = True
                        self.stats["crawled"] += 1
                        logger.info(f"✅ Episode {episode_num}: Crawled successfully")
                        break
                    else:
                        logger.warning(f"⚠️  Episode {episode_num}: Attempt {attempt + 1} failed")
                        
                except Exception as e:
                    logger.error(f"❌ Episode {episode_num}: Crawl error - {e}")
                
                if attempt < self.crawl_config.retry_count - 1:
                    logger.info(f"🔄 Retrying in {self.crawl_config.delay_between_episodes}s...")
                    time.sleep(self.crawl_config.delay_between_episodes)
            
            if success and video_path:
                crawled_videos.append({
                    "episode": episode_num,
                    "video_path": video_path,
                    "fps": episode_config.get("fps", 1.0)
                })
            else:
                logger.error(f"❌ Episode {episode_num}: Failed after {self.crawl_config.retry_count} attempts")
                self.stats["failed"] += 1
            
            # Delay between episodes
            if idx < len(batch) - 1:
                time.sleep(self.crawl_config.delay_between_episodes)
        
        # Phase 2: Ingest batch
        if crawled_videos:
            logger.info(f"\n💾 Phase 2/3: Ingesting {len(crawled_videos)} episodes...")
            
            for video_info in crawled_videos:
                episode_num = video_info["episode"]
                video_path = video_info["video_path"]
                fps = video_info["fps"]
                
                try:
                    logger.info(f"🔄 Ingesting Episode {episode_num}...")
                    
                    ingestor = AnimeIngestor(
                        video_path=video_path,
                        anime_id=anime_id,
                        episode=episode_num,
                        fps=fps,
                        season=metadata.season if metadata else None
                    )
                    
                    # Only pass metadata for first episode
                    if metadata and episode_num == crawled_videos[0]["episode"]:
                        ingestor.run(metadata)
                    else:
                        ingestor.run()
                    
                    self.stats["ingested"] += 1
                    logger.info(f"✅ Episode {episode_num}: Ingested successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Episode {episode_num}: Ingest failed - {e}")
                    self.stats["failed"] += 1
        
        # Phase 3: Cleanup batch
        if self.crawl_config.auto_cleanup and crawled_videos:
            logger.info(f"\n🧹 Phase 3/3: Cleaning up {len(crawled_videos)} videos...")
            
            for video_info in crawled_videos:
                video_path = video_info["video_path"]
                episode_num = video_info["episode"]
                
                try:
                    if os.path.exists(video_path):
                        file_size = os.path.getsize(video_path)
                        os.remove(video_path)
                        self.stats["cleaned_up"] += 1
                        self.stats["disk_space_freed"] += file_size
                        
                        size_mb = file_size / (1024 * 1024)
                        logger.info(f"🗑️  Episode {episode_num}: Deleted video ({size_mb:.1f} MB)")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Episode {episode_num}: Failed to delete video - {e}")
            
            freed_gb = self.stats["disk_space_freed"] / (1024 * 1024 * 1024)
            logger.info(f"✅ Batch cleanup complete! Freed: {freed_gb:.2f} GB")
    
    def _print_summary(self):
        """Print pipeline execution summary"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"Total Anime Processed: {self.stats['total_anime']}")
        logger.info(f"Total Episodes: {self.stats['total_episodes']}")
        logger.info(f"Successfully Crawled: {self.stats['crawled']}")
        logger.info(f"Successfully Ingested: {self.stats['ingested']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Videos Cleaned Up: {self.stats['cleaned_up']}")
        
        freed_gb = self.stats["disk_space_freed"] / (1024 * 1024 * 1024)
        logger.info(f"Disk Space Freed: {freed_gb:.2f} GB")
        
        success_rate = (self.stats['ingested'] / self.stats['total_episodes'] * 100) if self.stats['total_episodes'] > 0 else 0
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        logger.info("=" * 80)


def create_sample_crawl_config(output_path: str):
    """
    Tạo file config mẫu cho crawl pipeline
    
    Args:
        output_path: Path to save config file
    """
    sample_config = {
        "pipeline_settings": {
            "batch_size": 24,
            "auto_cleanup": True,
            "keep_frames": True,
            "retry_count": 3,
            "delay_between_episodes": 2
        },
        "anime": [
            {
                "anime_id": "jujutsu_kaisen_s2",
                "title": "Jujutsu Kaisen Season 2",
                "title_english": "Sorcery Fight Season 2",
                "title_japanese": "呪術廻戦 第2期",
                "title_vietnamese": "Chú Thuật Hồi Chiến Phần 2",
                "genres": ["Action", "Supernatural", "School"],
                "year": 2023,
                "rating": 8.8,
                "studio": "MAPPA",
                "studios": ["MAPPA"],
                "season": "2023-Summer",
                "description": "Sequel to Jujutsu Kaisen...",
                "episodes": [
                    {
                        "episode": 1,
                        "crawl_url": "https://vuighe.cam/chu-thuat-hoi-chien-phan-2/tap-1/",
                        "fps": 1.0
                    },
                    {
                        "episode": 2,
                        "crawl_url": "https://vuighe.cam/chu-thuat-hoi-chien-phan-2/tap-2/",
                        "fps": 1.0
                    }
                ]
            },
            {
                "anime_id": "one_piece",
                "title": "One Piece",
                "title_english": "One Piece",
                "title_japanese": "ワンピース",
                "title_vietnamese": "Vua Hải Tặc",
                "genres": ["Action", "Adventure", "Fantasy"],
                "year": 1999,
                "rating": 8.7,
                "studio": "Toei Animation",
                "studios": ["Toei Animation"],
                "season": "1999-Fall",
                "description": "Follows the adventures of Monkey D. Luffy...",
                "episodes": [
                    {
                        "episode": 1,
                        "video_path": "./data/videos/one_piece/ep001.mp4",
                        "fps": 1.0
                    }
                ]
            }
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Sample crawl config created at: {output_path}")
    logger.info("📝 Edit this file and add your anime crawl URLs")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Integrated crawl and ingest pipeline for anime"
    )
    parser.add_argument(
        "--config",
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--create-sample",
        help="Create sample config file at specified path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Number of episodes to process before cleanup (default: 24)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable auto cleanup of videos"
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="Number of retry attempts for failed episodes (default: 3)"
    )
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_crawl_config(args.create_sample)
    elif args.config:
        crawl_config = CrawlConfig(
            batch_size=args.batch_size,
            auto_cleanup=not args.no_cleanup,
            retry_count=args.retry
        )
        
        pipeline = IntegratedPipeline(args.config, crawl_config)
        pipeline.run()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
