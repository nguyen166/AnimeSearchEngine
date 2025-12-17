"""
VuiGhe Crawler
Specialized crawler for vuighe.cam anime website
"""

import json
import time
import logging
from typing import Optional, List

from selenium.webdriver.common.by import By

from .base import BaseCrawler, VideoMetadata

logger = logging.getLogger(__name__)


class VuiGheCrawler(BaseCrawler):
    """
    Crawler for vuighe.cam
    
    Features:
    - ArtPlayer video player support
    - Ad skipping with specific selector
    - M3U8 stream extraction
    """
    
    # Site-specific selectors
    TITLE_SELECTORS = [
        "h1.Title", ".Name", "h1.name",
        "h1.heading_movie", ".name-movie",
        ".movie-title", ".anime-title",
        "h1", "title"
    ]
    
    DESCRIPTION_SELECTORS = [
        ".Description", ".desc", ".description",
        ".movie-description", ".content-text",
        "p.story"
    ]
    
    AD_SKIP_SELECTORS = [
        ".artplayer-plugin-ads-close",  # Primary - VuiGhe uses ArtPlayer
        ".jw-skip",
        ".skip-ad", ".skip-button",
        "button[aria-label*='Skip']",
        ".ad-close"
    ]
    
    def __init__(self, output_dir: str = None, headless: bool = True):
        super().__init__(output_dir, headless)
        self.logger = logging.getLogger("VuiGheCrawler")
    
    def get_video_url(self) -> Optional[str]:
        """
        Extract M3U8/MP4 URL from VuiGhe player
        
        VuiGhe typically uses:
        - ArtPlayer with HLS streams
        - Multiple M3U8 URLs (ads + main video)
        - Longest URL is usually the main video
        
        Returns:
            Video URL if found
        """
        self.logger.info("🔍 Scanning network logs for video...")
        
        # Get all network URLs
        network_urls = self.get_network_urls(['.m3u8', '.mp4'])
        
        m3u8_urls = network_urls.get('.m3u8', [])
        mp4_urls = network_urls.get('.mp4', [])
        
        self.logger.info(f"📝 Found: {len(m3u8_urls)} M3U8, {len(mp4_urls)} MP4")
        
        # Prefer M3U8 (HLS streams)
        if m3u8_urls:
            # Choose longest URL (main video vs ads)
            # Main video URLs are typically longer due to more path segments
            video_url = max(m3u8_urls, key=len)
            self.logger.info(f"🎯 Selected M3U8 (longest of {len(m3u8_urls)})")
            return video_url
        
        # Fallback to MP4
        if mp4_urls:
            video_url = max(mp4_urls, key=len)
            self.logger.info(f"🎯 Selected MP4 (longest of {len(mp4_urls)})")
            return video_url
        
        return None
    
    def parse_page_info(self, url: str, anime_id: str, episode: int) -> VideoMetadata:
        """
        Parse VuiGhe page for metadata
        
        Args:
            url: Page URL
            anime_id: Anime identifier
            episode: Episode number
            
        Returns:
            VideoMetadata object
        """
        title = f"{anime_id} - Episode {episode}"
        description = ""
        
        # First, try to get title from <title> tag (most reliable)
        try:
            page_title = self.driver.title
            if page_title and len(page_title) > 3:
                # Clean up common suffixes like "- VuiGhe.cam", "| VuiGhe"
                title = page_title
                for suffix in [" - VuiGhe.cam", " | VuiGhe", " - VuiGhe", " Vietsub"]:
                    if suffix.lower() in title.lower():
                        idx = title.lower().find(suffix.lower())
                        title = title[:idx].strip()
                self.logger.info(f"🎬 Found title from <title>: {title}")
        except Exception as e:
            self.logger.debug(f"Could not get page title: {e}")
        
        # Fallback: Try CSS selectors for title (if <title> didn't work)
        if title == f"{anime_id} - Episode {episode}":
            for selector in self.TITLE_SELECTORS[:-1]:  # Skip 'title' selector
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in elements:
                        text = elem.text.strip()
                        if text and len(text) > 3:
                            title = text
                            self.logger.info(f"🎬 Found title via CSS: {title}")
                            break
                    if title != f"{anime_id} - Episode {episode}":
                        break
                except:
                    continue
        
        # Try to get description
        for selector in self.DESCRIPTION_SELECTORS:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elements:
                    text = elem.text.strip()
                    if text and len(text) > 20:
                        description = text[:500]  # Limit length
                        self.logger.info(f"📝 Found description: {description[:50]}...")
                        break
                if description:
                    break
            except:
                continue
        
        return VideoMetadata(
            anime_id=anime_id,
            title=title,
            episode=episode,
            description=description,
            source_url=url
        )
    
    def skip_ad(self, wait_time: int = 7) -> bool:
        """
        VuiGhe-specific ad skipping
        
        VuiGhe uses ArtPlayer with ads plugin.
        The skip button has class 'artplayer-plugin-ads-close' with text 'Đóng'
        """
        self.logger.info(f"⏳ Waiting {wait_time}s for ad to finish...")
        time.sleep(wait_time)
        
        # Try VuiGhe-specific selectors first
        for selector in self.AD_SKIP_SELECTORS:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elements:
                    if elem.is_displayed():
                        # Check for Vietnamese text "Đóng" (Close)
                        if "Đóng" in elem.text or "Close" in elem.text or elem.text == "":
                            elem.click()
                            self.logger.info(f"✅ Clicked skip ad: {selector}")
                            time.sleep(2)
                            return True
            except:
                continue
        
        # Fallback to base class method
        return super().skip_ad(wait_time=0)


# ============================================================================
# Convenience Functions
# ============================================================================

def crawl_vuighe(
    url: str,
    anime_id: str,
    episode: int,
    season: str = "",
    output_dir: str = "./data/raw_videos",
    headless: bool = True
):
    """
    Convenience function to crawl a single VuiGhe episode
    
    Args:
        url: Episode URL
        anime_id: Anime identifier
        episode: Episode number
        season: Season identifier (optional)
        output_dir: Output directory
        headless: Run headless
        
    Returns:
        CrawlResult
    """
    crawler = VuiGheCrawler(output_dir=output_dir, headless=headless)
    return crawler.crawl(url, anime_id, episode, season)


def crawl_vuighe_batch(
    episodes: List[dict],
    output_dir: str = "./data/raw_videos",
    headless: bool = True,
    delay: int = 5
):
    """
    Convenience function to crawl multiple VuiGhe episodes
    
    Args:
        episodes: List of dicts with keys: url, anime_id, episode, season
        output_dir: Output directory
        headless: Run headless
        delay: Seconds between episodes
        
    Returns:
        List of CrawlResult
    """
    crawler = VuiGheCrawler(output_dir=output_dir, headless=headless)
    return crawler.crawl_batch(episodes, delay)
