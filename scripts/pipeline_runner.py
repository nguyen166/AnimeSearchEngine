"""
Pipeline Runner
Script để chạy tự động ingestion cho nhiều tập anime
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.ingest_anime import AnimeIngestor
from app.models.schemas import AnimeMetadata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """Class để chạy ingestion pipeline cho nhiều videos"""
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline runner
        
        Args:
            config_path: Path to JSON config file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration từ JSON file"""
        logger.info(f"Loading config from {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config
    
    def run(self):
        """Run pipeline cho tất cả anime trong config"""
        logger.info("Starting pipeline runner...")
        
        total_anime = len(self.config.get("anime", []))
        logger.info(f"Found {total_anime} anime in config")
        
        success_count = 0
        failed_count = 0
        
        for anime_config in self.config["anime"]:
            try:
                self._process_anime(anime_config)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to process anime: {e}")
                failed_count += 1
        
        logger.info("=" * 60)
        logger.info(f"Pipeline completed!")
        logger.info(f"Success: {success_count}, Failed: {failed_count}")
        logger.info("=" * 60)
    
    def _process_anime(self, anime_config: Dict[str, Any]):
        """
        Process một anime
        
        Args:
            anime_config: Anime configuration
        """
        anime_id = anime_config["anime_id"]
        title = anime_config["title"]
        
        logger.info("=" * 60)
        logger.info(f"Processing: {title} ({anime_id})")
        logger.info("=" * 60)
        
        # Create metadata
        metadata = AnimeMetadata(
            anime_id=anime_id,
            title=title,
            title_english=anime_config.get("title_english"),
            title_japanese=anime_config.get("title_japanese"),
            genres=anime_config.get("genres", []),
            year=anime_config.get("year"),
            episodes=len(anime_config.get("episodes", [])),
            rating=anime_config.get("rating"),
            description=anime_config.get("description"),
            studio=anime_config.get("studio")
        )
        
        # Process each episode
        episodes = anime_config.get("episodes", [])
        total_episodes = len(episodes)
        
        logger.info(f"Found {total_episodes} episodes")
        
        for idx, episode_config in enumerate(episodes, 1):
            episode_num = episode_config.get("episode", idx)
            video_path = episode_config["video_path"]
            fps = episode_config.get("fps", 1.0)
            
            logger.info(f"Processing episode {episode_num}/{total_episodes}")
            logger.info(f"Video: {video_path}")
            
            # Check if video exists
            if not os.path.exists(video_path):
                logger.error(f"Video not found: {video_path}")
                continue
            
            # Create ingestor
            ingestor = AnimeIngestor(
                video_path=video_path,
                anime_id=anime_id,
                episode=episode_num,
                fps=fps
            )
            
            # Run ingestion (only pass metadata for first episode)
            if idx == 1:
                ingestor.run(metadata)
            else:
                ingestor.run()
            
            logger.info(f"✓ Episode {episode_num} completed")
        
        logger.info(f"✓ All episodes of {title} completed")


def create_sample_config(output_path: str):
    """
    Tạo file config mẫu
    
    Args:
        output_path: Path to save config file
    """
    sample_config = {
        "anime": [
            {
                "anime_id": "one_piece_001",
                "title": "One Piece",
                "title_english": "One Piece",
                "title_japanese": "ワンピース",
                "genres": ["Action", "Adventure", "Fantasy"],
                "year": 1999,
                "rating": 8.7,
                "studio": "Toei Animation",
                "description": "Follows the adventures of Monkey D. Luffy and his pirate crew...",
                "episodes": [
                    {
                        "episode": 1,
                        "video_path": "./data/videos/one_piece/ep001.mp4",
                        "fps": 1.0
                    },
                    {
                        "episode": 2,
                        "video_path": "./data/videos/one_piece/ep002.mp4",
                        "fps": 1.0
                    }
                ]
            },
            {
                "anime_id": "naruto_001",
                "title": "Naruto",
                "title_english": "Naruto",
                "title_japanese": "ナルト",
                "genres": ["Action", "Adventure", "Martial Arts"],
                "year": 2002,
                "rating": 8.3,
                "studio": "Studio Pierrot",
                "description": "Naruto Uzumaki, a young ninja...",
                "episodes": [
                    {
                        "episode": 1,
                        "video_path": "./data/videos/naruto/ep001.mp4",
                        "fps": 1.0
                    }
                ]
            }
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sample config created at: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run ingestion pipeline for multiple anime"
    )
    parser.add_argument(
        "--config",
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--create-sample",
        help="Create sample config file at specified path"
    )
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_config(args.create_sample)
    elif args.config:
        runner = PipelineRunner(args.config)
        runner.run()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
