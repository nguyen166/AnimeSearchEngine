#!/usr/bin/env python3
"""
Crawler Runner Script
Main entry point for crawling anime videos
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from crawlers import VuiGheCrawler, CrawlResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('crawler.log')
    ]
)
logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_OUTPUT_DIR = "./data/raw_videos"


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load crawl configuration from JSON file
    
    Expected format:
    {
        "output_dir": "./data/raw_videos",
        "headless": true,
        "delay_between_episodes": 5,
        "anime": [
            {
                "anime_id": "jujutsu-kaisen-s2",
                "season": "S2",
                "episodes": [
                    {"episode": 1, "url": "https://vuighe.cam/..."},
                    {"episode": 2, "url": "https://vuighe.cam/..."}
                ]
            }
        ]
    }
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_sample_config(output_path: str):
    """Create sample configuration file"""
    sample = {
        "output_dir": "./data/raw_videos",
        "headless": True,
        "delay_between_episodes": 5,
        "anime": [
            {
                "anime_id": "jujutsu-kaisen-s2",
                "title": "Jujutsu Kaisen Season 2",
                "season": "S2",
                "episodes": [
                    {
                        "episode": 1,
                        "url": "https://vuighe.cam/chu-thuat-hoi-chien-phan-2/tap-1/"
                    },
                    {
                        "episode": 2,
                        "url": "https://vuighe.cam/chu-thuat-hoi-chien-phan-2/tap-2/"
                    }
                ]
            }
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📄 Sample config created: {output_path}")
    return output_path


def crawl_from_config(config: Dict[str, Any]) -> List[CrawlResult]:
    """
    Crawl episodes based on configuration
    
    Args:
        config: Configuration dict
        
    Returns:
        List of CrawlResult
    """
    output_dir = config.get('output_dir', DEFAULT_OUTPUT_DIR)
    headless = config.get('headless', True)
    delay = config.get('delay_between_episodes', 5)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for anime in config.get('anime', []):
        anime_id = anime['anime_id']
        season = anime.get('season', '')
        episodes = anime.get('episodes', [])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📺 Anime: {anime.get('title', anime_id)}")
        logger.info(f"   ID: {anime_id}, Season: {season}")
        logger.info(f"   Episodes to crawl: {len(episodes)}")
        logger.info(f"{'='*60}")
        
        # Build episode list for batch crawl
        episode_list = []
        for ep in episodes:
            episode_list.append({
                'url': ep['url'],
                'anime_id': anime_id,
                'episode': ep['episode'],
                'season': season
            })
        
        # Crawl with VuiGhe crawler
        crawler = VuiGheCrawler(output_dir=output_dir, headless=headless)
        results = crawler.crawl_batch(episode_list, delay=delay)
        all_results.extend(results)
    
    return all_results


def crawl_single(
    url: str,
    anime_id: str,
    episode: int,
    season: str = "",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    headless: bool = True
) -> CrawlResult:
    """
    Crawl a single episode
    
    Args:
        url: Episode URL
        anime_id: Anime identifier  
        episode: Episode number
        season: Season identifier
        output_dir: Output directory
        headless: Run headless
        
    Returns:
        CrawlResult
    """
    crawler = VuiGheCrawler(output_dir=output_dir, headless=headless)
    return crawler.crawl(url, anime_id, episode, season)


def print_summary(results: List[CrawlResult]):
    """Print crawl summary"""
    total = len(results)
    success = sum(1 for r in results if r.success)
    failed = total - success
    
    logger.info(f"\n{'='*60}")
    logger.info("📊 CRAWL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"   Total: {total}")
    logger.info(f"   ✅ Success: {success}")
    logger.info(f"   ❌ Failed: {failed}")
    
    if failed > 0:
        logger.info("\n   Failed episodes:")
        for i, r in enumerate(results):
            if not r.success:
                logger.info(f"   - Episode {i+1}: {r.error}")
    
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Anime Video Crawler - Download anime episodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample config
  python run_crawler.py --create-sample my_config.json
  
  # Crawl from config file
  python run_crawler.py --config my_config.json
  
  # Crawl single episode
  python run_crawler.py --url "https://vuighe.cam/..." --anime-id "naruto" --episode 1
  
  # Crawl with visible browser (for debugging)
  python run_crawler.py --url "https://..." --anime-id "test" --episode 1 --no-headless
        """
    )
    
    # Config file mode
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to JSON config file'
    )
    
    parser.add_argument(
        '--create-sample',
        type=str,
        metavar='PATH',
        help='Create sample config file at specified path'
    )
    
    # Single episode mode
    parser.add_argument(
        '--url', '-u',
        type=str,
        help='URL of episode to crawl'
    )
    
    parser.add_argument(
        '--anime-id', '-a',
        type=str,
        help='Anime identifier'
    )
    
    parser.add_argument(
        '--episode', '-e',
        type=int,
        help='Episode number'
    )
    
    parser.add_argument(
        '--season', '-s',
        type=str,
        default='',
        help='Season identifier (optional)'
    )
    
    # Common options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--no-headless',
        action='store_true',
        help='Show browser window (useful for debugging)'
    )
    
    args = parser.parse_args()
    
    # Create sample config
    if args.create_sample:
        create_sample_config(args.create_sample)
        return
    
    # Crawl from config file
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"❌ Config file not found: {args.config}")
            sys.exit(1)
        
        config = load_config(args.config)
        
        # Override config with CLI args if provided
        if args.output_dir != DEFAULT_OUTPUT_DIR:
            config['output_dir'] = args.output_dir
        if args.no_headless:
            config['headless'] = False
        
        results = crawl_from_config(config)
        print_summary(results)
        
        # Exit with error code if any failed
        if any(not r.success for r in results):
            sys.exit(1)
        return
    
    # Single episode mode
    if args.url and args.anime_id and args.episode:
        result = crawl_single(
            url=args.url,
            anime_id=args.anime_id,
            episode=args.episode,
            season=args.season,
            output_dir=args.output_dir,
            headless=not args.no_headless
        )
        
        if result.success:
            logger.info(f"✅ Success! Video: {result.video_path}")
            logger.info(f"   Metadata: {result.metadata_path}")
        else:
            logger.error(f"❌ Failed: {result.error}")
            sys.exit(1)
        return
    
    # No valid arguments
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
