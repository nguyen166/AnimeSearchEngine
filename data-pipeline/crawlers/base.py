"""
Base Crawler Module
Abstract base class for all anime crawlers
"""

import os
import sys
import json
import time
import logging
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

# Import selenium_stealth
try:
    from selenium_stealth import stealth
    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False
    print("⚠️ selenium-stealth not installed. Run: pip install selenium-stealth")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VideoMetadata:
    """Metadata for a crawled video"""
    anime_id: str
    title: str
    episode: int
    season: str = ""
    description: str = ""
    source_url: str = ""
    video_path: str = ""
    duration: Optional[float] = None
    file_size: Optional[int] = None
    crawled_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save_json(self, output_path: str):
        """Save metadata to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"📄 Metadata saved: {output_path}")
    
    @classmethod
    def from_json(cls, json_path: str) -> 'VideoMetadata':
        """Load metadata from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class CrawlResult:
    """Result of a crawl operation"""
    success: bool
    video_path: Optional[str] = None
    metadata_path: Optional[str] = None
    metadata: Optional[VideoMetadata] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'success': self.success,
            'video_path': self.video_path,
            'metadata_path': self.metadata_path,
            'error': self.error
        }
        if self.metadata:
            result['metadata'] = self.metadata.to_dict()
        return result


# ============================================================================
# Base Crawler Class
# ============================================================================

class BaseCrawler(ABC):
    """
    Abstract base class for anime crawlers
    
    Subclasses must implement:
    - get_video_url(): Extract video URL from page
    - parse_page_info(): Parse anime metadata from page
    """
    
    # Default output directory
    DEFAULT_OUTPUT_DIR = "./data/raw_videos"
    
    # URL blocklist for network log filtering
    URL_BLOCKLIST = [
        ".gif", ".png", ".jpg", ".jpeg", ".webp", ".svg",
        "google-analytics", "facebook", "ping", "stats",
        "jwpltx", "gstatic", "favicon", "pixel", "tracking",
        "analytics", "ads", "advertisement"
    ]
    
    def __init__(self, output_dir: str = None, headless: bool = True):
        """
        Initialize crawler
        
        Args:
            output_dir: Directory to save videos and metadata
            headless: Run browser in headless mode
        """
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        self.headless = headless
        self.driver: Optional[webdriver.Chrome] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    # ========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # ========================================================================
    
    @abstractmethod
    def get_video_url(self) -> Optional[str]:
        """
        Extract video URL (M3U8/MP4) from current page
        
        Returns:
            Video URL if found, None otherwise
        """
        pass
    
    @abstractmethod
    def parse_page_info(self, url: str, anime_id: str, episode: int) -> VideoMetadata:
        """
        Parse page to extract video metadata
        
        Args:
            url: Page URL
            anime_id: Anime identifier
            episode: Episode number
            
        Returns:
            VideoMetadata object
        """
        pass
    
    # ========================================================================
    # Selenium WebDriver Management
    # ========================================================================
    
    def setup_driver(self) -> webdriver.Chrome:
        """
        Setup Chrome WebDriver with anti-detection
        
        Returns:
            Configured Chrome WebDriver
        """
        options = webdriver.ChromeOptions()
        
        # Basic options
        if self.headless:
            options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        # Anti-detection
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Enable performance logging for network requests
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
        
        driver = webdriver.Chrome(options=options)
        
        # Apply stealth mode if available
        if HAS_STEALTH:
            stealth(driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
            )
        
        self.driver = driver
        return driver
    
    def close_driver(self):
        """Close WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    # ========================================================================
    # Common Crawling Utilities
    # ========================================================================
    
    def bypass_cloudflare(self, max_wait: int = 10) -> bool:
        """
        Attempt to bypass Cloudflare challenge
        
        Args:
            max_wait: Maximum seconds to wait
            
        Returns:
            True if bypassed or not blocked
        """
        self.logger.info("🛡️ Checking for Cloudflare...")
        time.sleep(5)
        
        title = self.driver.title.lower()
        blocking_keywords = ["just a moment", "xác minh", "security", "checking"]
        
        if not any(kw in title for kw in blocking_keywords):
            self.logger.info("✅ No Cloudflare blocking detected")
            return True
        
        try:
            # Find and interact with challenge iframe
            iframes = self.driver.find_elements(
                By.XPATH, 
                "//iframe[contains(@src, 'cloudflare') or contains(@src, 'turnstile') or contains(@title, 'widget')]"
            )
            
            if iframes:
                self.logger.info("⚠️ Cloudflare iframe detected, attempting bypass...")
                self.driver.switch_to.frame(iframes[0])
                
                try:
                    checkbox = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((
                            By.CSS_SELECTOR, 
                            "input[type='checkbox'], .ctp-checkbox-label, #challenge-stage, body"
                        ))
                    )
                    checkbox.click()
                    self.logger.info("✅ Clicked Cloudflare checkbox")
                except:
                    action = ActionChains(self.driver)
                    action.move_by_offset(50, 50).click().perform()
                
                self.driver.switch_to.default_content()
                time.sleep(max_wait)
                return True
            
            # Try challenge stage button
            verify_btn = self.driver.find_elements(By.ID, "challenge-stage")
            if verify_btn:
                verify_btn[0].click()
                time.sleep(max_wait)
                return True
                
        except Exception as e:
            self.logger.debug(f"Bypass attempt: {e}")
            self.driver.switch_to.default_content()
        
        return False
    
    def click_play_button(self):
        """Try to click video player to trigger loading"""
        play_selectors = [
            ".jw-display-icon-container",
            ".vjs-big-play-button",
            "#play-button",
            ".play-icon",
            "div.player-wrapper",
            "div[class*='player']",
            ".play-btn",
            "[aria-label='Play']"
        ]
        
        self.logger.info("🖱️ Attempting to interact with player...")
        actions = ActionChains(self.driver)
        
        # Scroll to trigger lazy load
        self.driver.execute_script("window.scrollTo(0, 300);")
        time.sleep(1)
        
        # Click center of screen (fallback)
        try:
            body = self.driver.find_element(By.TAG_NAME, "body")
            actions.move_to_element_with_offset(body, 500, 350).click().perform()
        except:
            pass
        
        for selector in play_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elements:
                    if elem.is_displayed():
                        actions.move_to_element(elem).click().perform()
                        self.logger.info(f"👉 Clicked: {selector}")
                        time.sleep(1)
                        return True
            except:
                continue
        
        return False
    
    def skip_ad(self, wait_time: int = 5) -> bool:
        """
        Wait for and click skip ad button
        
        Args:
            wait_time: Seconds to wait before looking for skip button
            
        Returns:
            True if ad was skipped
        """
        self.logger.info(f"⏳ Waiting {wait_time}s for skip ad button...")
        time.sleep(wait_time)
        
        skip_selectors = [
            ".artplayer-plugin-ads-close",  # VuiGhe ArtPlayer
            ".jw-skip",  # JW Player
            ".skip-ad", ".skip-button",
            "button[aria-label*='Skip']",
            ".ad-close", ".close-ad",
            ".ytp-ad-skip-button",  # YouTube-style
            "div[class*='skip']",
            "button[class*='skip']",
            ".video-ads-skip",
        ]
        
        for selector in skip_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elements:
                    if elem.is_displayed():
                        elem.click()
                        self.logger.info(f"✅ Clicked skip ad: {selector}")
                        time.sleep(2)
                        return True
            except:
                continue
        
        # Try by text content
        skip_texts = ["Skip", "Bỏ qua", "Close", "Đóng", "X"]
        for text in skip_texts:
            try:
                buttons = self.driver.find_elements(
                    By.XPATH, 
                    f"//button[contains(text(), '{text}')] | //div[contains(text(), '{text}') and contains(@class, 'skip')]"
                )
                for btn in buttons:
                    if btn.is_displayed():
                        btn.click()
                        self.logger.info(f"✅ Clicked skip by text: {text}")
                        time.sleep(2)
                        return True
            except:
                pass
        
        self.logger.info("ℹ️ No skip ad button found")
        return False
    
    def get_network_urls(self, extensions: List[str] = None) -> Dict[str, List[str]]:
        """
        Extract URLs from network logs by extension
        
        Args:
            extensions: List of file extensions to look for
            
        Returns:
            Dict mapping extension to list of URLs
        """
        if extensions is None:
            extensions = ['.m3u8', '.mp4', '.webm']
        
        logs = self.driver.get_log("performance")
        urls = {ext: [] for ext in extensions}
        
        for entry in logs:
            try:
                message = json.loads(entry["message"])["message"]
                if message["method"] == "Network.requestWillBeSent":
                    url = message["params"]["request"]["url"]
                    
                    # Skip blocklisted URLs
                    if any(blocked in url.lower() for blocked in self.URL_BLOCKLIST):
                        continue
                    
                    for ext in extensions:
                        if ext in url.lower():
                            urls[ext].append(url)
            except:
                continue
        
        return urls
    
    # ========================================================================
    # Download Methods
    # ========================================================================
    
    def download_with_ffmpeg(
        self, 
        url: str, 
        output_path: str, 
        referer_url: str
    ) -> Optional[str]:
        """
        Download video using FFmpeg
        
        Args:
            url: Video URL
            output_path: Output file path
            referer_url: Referer URL for headers
            
        Returns:
            Output path if successful, None otherwise
        """
        self.logger.info(f"⬇️ Downloading: {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        parsed = urlparse(referer_url)
        origin = f"{parsed.scheme}://{parsed.netloc}/"
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        
        cmd = [
            "ffmpeg", "-y",
            "-user_agent", user_agent,
            "-headers", f"Referer: {referer_url}",
            "-headers", f"Origin: {origin}",
            "-reconnect", "1",
            "-reconnect_at_eof", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "2",
            "-protocol_whitelist", "file,http,https,tcp,tls,crypto",
            "-i", url,
            "-c", "copy",
            "-bsf:a", "aac_adtstoasc",
            "-loglevel", "error",
            output_path
        ]
        
        try:
            with open(os.devnull, 'w') as devnull:
                subprocess.run(
                    cmd, 
                    check=True,
                    stdout=devnull,  # "Hố đen" cho log thông thường
                    stderr=devnull   # "Hố đen" cho log lỗi (FFmpeg ghi log vào stderr)
                )
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 500000:
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                self.logger.info(f"✅ Downloaded: {size_mb:.2f} MB")
                return output_path
            else:
                self.logger.error("❌ Download failed: File too small or empty")
                return None
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ FFmpeg error: {e}")
            return None
        except FileNotFoundError:
            self.logger.error("❌ FFmpeg not found! Please install FFmpeg")
            return None
    
    # ========================================================================
    # Debug Utilities
    # ========================================================================
    
    def save_debug_info(self, prefix: str = "debug"):
        """Save screenshot and HTML for debugging"""
        try:
            debug_dir = os.path.join(self.output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = int(time.time())
            screenshot_path = os.path.join(debug_dir, f"{prefix}_{timestamp}.png")
            html_path = os.path.join(debug_dir, f"{prefix}_{timestamp}.html")
            
            self.driver.save_screenshot(screenshot_path)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            
            self.logger.info(f"📸 Debug saved: {screenshot_path}")
        except Exception as e:
            self.logger.error(f"Failed to save debug info: {e}")
    
    # ========================================================================
    # Main Crawl Method
    # ========================================================================
    
    def crawl(
        self,
        url: str,
        anime_id: str,
        episode: int,
        season: str = ""
    ) -> CrawlResult:
        """
        Crawl a single episode
        
        Args:
            url: Page URL
            anime_id: Anime identifier
            episode: Episode number
            season: Season identifier (optional)
            
        Returns:
            CrawlResult with video path and metadata
        """
        try:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🎬 Crawling: {anime_id} Episode {episode}")
            self.logger.info(f"🔗 URL: {url}")
            self.logger.info(f"{'='*60}")
            
            # Setup driver
            self.setup_driver()
            self.driver.get(url)
            time.sleep(5)
            
            # Bypass protection
            self.bypass_cloudflare()
            
            # Interact with player
            self.click_play_button()
            self.skip_ad()
            
            # Wait for video to load
            self.logger.info("⏳ Waiting for video to load...")
            time.sleep(15)
            
            # Get video URL
            video_url = self.get_video_url()
            
            if not video_url:
                self.logger.error("❌ Video URL not found!")
                self.save_debug_info("no_video")
                return CrawlResult(success=False, error="Video URL not found")
            
            self.logger.info(f"🔗 Video URL: {video_url[:80]}...")
            
            # Generate filenames (consistent naming for ingest_video.py)
            # Format: {anime_id}_ep{episode:03d}.mp4 / .json
            safe_anime_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in anime_id)
            base_filename = f"{safe_anime_id}_ep{episode:03d}"
            video_path = os.path.join(self.output_dir, f"{base_filename}.mp4")
            metadata_path = os.path.join(self.output_dir, f"{base_filename}.json")
            
            # Parse metadata from page
            metadata = self.parse_page_info(url, anime_id, episode)
            metadata.season = season
            metadata.source_url = url
            metadata.video_path = video_path
            
            # Download video
            result = self.download_with_ffmpeg(video_url, video_path, url)
            
            if not result:
                self.save_debug_info("download_failed")
                return CrawlResult(success=False, error="Download failed")
            
            # Update metadata with file info
            metadata.file_size = os.path.getsize(video_path)
            
            # Save metadata JSON
            metadata.save_json(metadata_path)
            
            self.logger.info(f"✅ Success! Video: {video_path}")
            
            return CrawlResult(
                success=True,
                video_path=video_path,
                metadata_path=metadata_path,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"❌ Crawl error: {e}", exc_info=True)
            if self.driver:
                self.save_debug_info("exception")
            return CrawlResult(success=False, error=str(e))
            
        finally:
            self.close_driver()
    
    def crawl_batch(
        self,
        episodes: List[Dict[str, Any]],
        delay: int = 5
    ) -> List[CrawlResult]:
        """
        Crawl multiple episodes
        
        Args:
            episodes: List of dicts with keys: url, anime_id, episode, season (optional)
            delay: Seconds to wait between episodes
            
        Returns:
            List of CrawlResult
        """
        results = []
        total = len(episodes)
        
        for i, ep in enumerate(episodes, 1):
            self.logger.info(f"\n📺 Progress: {i}/{total}")
            
            result = self.crawl(
                url=ep['url'],
                anime_id=ep['anime_id'],
                episode=ep['episode'],
                season=ep.get('season', '')
            )
            results.append(result)
            
            if i < total:
                self.logger.info(f"⏳ Waiting {delay}s before next episode...")
                time.sleep(delay)
        
        # Summary
        success_count = sum(1 for r in results if r.success)
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 Batch Complete: {success_count}/{total} successful")
        self.logger.info(f"{'='*60}")
        
        return results
