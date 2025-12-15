"""
Anime Video Crawler
Crawl anime videos from various websites using Selenium + FFmpeg
"""

import sys
import os
import time
import json
import re
import subprocess
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

# Import selenium_stealth
try:
    from selenium_stealth import stealth
except ImportError:
    print("⚠️ selenium-stealth not installed!")
    print("👉 Install: pip install selenium-stealth")
    stealth = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_driver(headless: bool = True) -> webdriver.Chrome:
    """
    Setup Chrome WebDriver with stealth mode
    
    Args:
        headless: Run in headless mode
        
    Returns:
        Configured Chrome WebDriver
    """
    options = webdriver.ChromeOptions()

    # Basic options
    if headless:
        options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-blink-features=AutomationControlled')
    
    # Anti-detection
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--disable-blink-features=AutomationControlled')
    
    # Enable performance logging
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    driver = webdriver.Chrome(options=options)

    # Apply stealth if available
    if stealth:
        stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )

    return driver


def bypass_cloudflare(driver) -> bool:
    """
    Try to bypass Cloudflare challenge
    
    Args:
        driver: Selenium WebDriver
        
    Returns:
        True if bypassed or not blocked, False otherwise
    """
    logger.info("🛡️ Checking for Cloudflare...")
    time.sleep(5)  # Wait for page load

    # Check page title for blocking indicators
    title = driver.title.lower()
    if "just a moment" not in title and "xác minh" not in title and "security" not in title:
        logger.info("✅ No blocking detected")
        return True

    try:
        # Method 1: Find iframe containing checkbox (Turnstile/Challenge)
        iframes = driver.find_elements(By.XPATH, "//iframe[contains(@src, 'cloudflare') or contains(@src, 'turnstile') or contains(@title, 'widget')]")

        if iframes:
            logger.info("⚠️ Cloudflare iframe detected, attempting click...")
            driver.switch_to.frame(iframes[0])

            try:
                checkbox = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='checkbox'], .ctp-checkbox-label, #challenge-stage, body"))
                )
                checkbox.click()
                logger.info("✅ Clicked Cloudflare checkbox")
            except:
                action = ActionChains(driver)
                action.move_by_offset(50, 50).click().perform()
                logger.info("⚡ Random click in iframe")

            driver.switch_to.default_content()
            time.sleep(8)
            return True

        # Method 2: Find verify button in main page
        verify_btn = driver.find_elements(By.ID, "challenge-stage")
        if verify_btn:
            logger.info("⚠️ Challenge stage detected, clicking...")
            verify_btn[0].click()
            time.sleep(8)
            return True

    except Exception as e:
        logger.info(f"ℹ️ Bypass error (may have passed automatically): {str(e)[:50]}")
        driver.switch_to.default_content()

    return False


def click_play_button(driver):
    """
    Try to find and click play button
    
    Args:
        driver: Selenium WebDriver
    """
    try:
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

        logger.info("🖱️ Trying to interact with player...")
        actions = ActionChains(driver)

        # Scroll down slightly to activate lazy load
        driver.execute_script("window.scrollTo(0, 300);")
        time.sleep(1)

        # Click center of screen (fallback)
        try:
            body = driver.find_element(By.TAG_NAME, "body")
            actions.move_to_element_with_offset(body, 500, 350).click().perform()
        except:
            pass

        for selector in play_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elements:
                    if elem.is_displayed():
                        actions.move_to_element(elem).click().perform()
                        logger.info(f"👉 Clicked: {selector}")
                        time.sleep(1)
            except:
                continue

    except:
        pass


def skip_ad(driver):
    """
    Try to find and click skip ad button after 5 seconds
    
    Args:
        driver: Selenium WebDriver
    """
    try:
        logger.info("⏳ Waiting 5s for skip ad button...")
        time.sleep(5)
        
        # Primary selector for vuighe.cam (ArtPlayer plugin)
        skip_selectors = [
            ".artplayer-plugin-ads-close",  # VuiGhe ArtPlayer ads close button (text: "Đóng")
            ".jw-skip",  # JW Player skip button
            ".skip-ad", ".skip-button",
            "button[aria-label*='Skip']",
            ".ad-close", ".close-ad",
            ".ytp-ad-skip-button",  # YouTube-style players
            "div[class*='skip']",
            "button[class*='skip']",
            ".video-ads-skip",
        ]
        
        logger.info("🔍 Looking for skip ad button...")
        actions = ActionChains(driver)
        
        for selector in skip_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elements:
                    if elem.is_displayed():
                        elem.click()
                        logger.info(f"✅ Clicked skip ad button: {selector}")
                        time.sleep(2)
                        return True
            except:
                continue
        
        # Try clicking by text content
        try:
            skip_texts = ["Skip", "Bỏ qua", "Close", "Đóng", "X"]
            for text in skip_texts:
                buttons = driver.find_elements(By.XPATH, f"//button[contains(text(), '{text}')] | //div[contains(text(), '{text}') and contains(@class, 'skip')]")
                for btn in buttons:
                    if btn.is_displayed():
                        btn.click()
                        logger.info(f"✅ Clicked skip ad by text: {text}")
                        time.sleep(2)
                        return True
        except:
            pass
        
        logger.info("ℹ️ No skip ad button found (may not have ads)")
        return False
        
    except:
        pass


def get_m3u8_link(driver, output_dir: str = "./data/videos") -> Optional[str]:
    """
    Find video link in network logs
    
    Args:
        driver: Selenium WebDriver
        output_dir: Output directory for videos (used to determine debug folder)
        
    Returns:
        M3U8 or MP4 URL if found, None otherwise
    """
    logger.info("🔍 Scanning network logs for video...")

    bypass_cloudflare(driver)

    # Check if still blocked
    if "just a moment" in driver.title.lower():
        logger.warning("❌ Still blocked by Cloudflare, trying refresh...")
        driver.refresh()
        time.sleep(5)
        bypass_cloudflare(driver)

    click_play_button(driver)
    
    # Skip ads if present (wait 5s then click skip button)
    skip_ad(driver)
    
    # Wait longer for main video to load after skipping ad
    logger.info("⏳ Waiting for main video to load (15s)...")
    time.sleep(15)

    logs = driver.get_log("performance")
    m3u8_candidates = []
    mp4_candidates = []

    # Blocklist - simple and effective
    blocklist = [".gif", ".png", ".jpg", "google-analytics", "facebook", "ping", "stats", "jwpltx", "gstatic", "favicon", "pixel"]

    for entry in logs:
        try:
            message = json.loads(entry["message"])["message"]
            if message["method"] == "Network.requestWillBeSent":
                url = message["params"]["request"]["url"]

                if any(bad in url for bad in blocklist):
                    continue

                if ".m3u8" in url:
                    m3u8_candidates.append(url)
                    logger.info(f"📝 Collected M3U8: {url[:80]}...")

                if ".mp4" in url:
                    mp4_candidates.append(url)
        except:
            continue

    # Choose the LONGEST M3U8 URL (main video is usually longer than ads)
    if m3u8_candidates:
        video_url = max(m3u8_candidates, key=len)
        logger.info(f"🎯 Selected M3U8 ({len(m3u8_candidates)} total, chose longest): {video_url[:80]}...")
        return video_url

    # Fallback to longest MP4
    if mp4_candidates:
        video_url = max(mp4_candidates, key=len)
        logger.info(f"🎯 Selected MP4 ({len(mp4_candidates)} total, chose longest): {video_url[:80]}...")
        return video_url
    
    return None


def download_with_ffmpeg(url: str, output_path: str, referer_url: str) -> Optional[str]:
    """
    Download video using FFmpeg with headers
    
    Args:
        url: Video URL (M3U8 or MP4)
        output_path: Output file path
        referer_url: Referer URL for headers
        
    Returns:
        Output path if successful, None otherwise
    """
    logger.info(f"⬇️ Downloading video to: {output_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    parsed_uri = urlparse(referer_url)
    domain = f"{parsed_uri.scheme}://{parsed_uri.netloc}/"

    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    cmd = [
        "ffmpeg", "-y",
        "-user_agent", user_agent,
        "-headers", f"Referer: {referer_url}",
        "-headers", f"Origin: {domain}",
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
        subprocess.run(cmd, check=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 500000:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"✅ Download successful! Size: {size_mb:.2f} MB")
            return output_path
        else:
            logger.error("❌ Download failed: Empty file")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FFmpeg error: {e}")
        return None
    except FileNotFoundError:
        logger.error("❌ FFmpeg not found! Please install FFmpeg")
        return None


def save_debug_info(driver, output_dir: str, step: str = "error"):
    """
    Save debug information (screenshot + HTML)
    
    Args:
        driver: Selenium WebDriver
        output_dir: Output directory
        step: Debug step name
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        screenshot_path = os.path.join(output_dir, f"debug_{step}_{int(time.time())}.png")
        html_path = os.path.join(output_dir, f"debug_{step}_{int(time.time())}.html")
        
        driver.save_screenshot(screenshot_path)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        
        logger.info(f"📸 Debug info saved: {screenshot_path}")
    except Exception as e:
        logger.error(f"Failed to save debug info: {e}")


def crawl_episode(
    url: str,
    anime_id: str,
    episode: int,
    output_dir: str = "./data/videos",
    headless: bool = True
) -> Optional[str]:
    """
    Crawl một episode anime từ URL
    
    Args:
        url: URL của trang chứa video
        anime_id: ID của anime
        episode: Số tập
        output_dir: Thư mục lưu video
        headless: Chạy ở chế độ headless
        
    Returns:
        Path to downloaded video if successful, None otherwise
    """
    driver = None
    
    try:
        logger.info(f"🌍 Accessing: {url}")
        
        # Setup driver
        driver = setup_driver(headless=headless)
        driver.get(url)
        time.sleep(5)

        # Get video title
        # Convert episode to int if it's a string
        episode_num = int(episode) if isinstance(episode, str) else episode
        title = f"{anime_id}_ep{episode_num:03d}"
        try:
            selectors = [
                "h1.Title", ".Name", "h1.name",  # AnimeVietsub
                "h1.heading_movie", ".name-movie",  # AnimeHay
                "h1", "title"
            ]
            for sel in selectors:
                elems = driver.find_elements(By.CSS_SELECTOR, sel)
                if elems and len(elems[0].text) > 2:
                    detected_title = elems[0].text.strip()
                    logger.info(f"🎬 Detected title: {detected_title}")
                    break
        except:
            pass

        # Generate safe filename
        safe_filename = re.sub(r'[\\/*?:"<>|]', "", title) + ".mp4"
        output_path = os.path.join(output_dir, safe_filename)

        # Get M3U8 link
        m3u8_url = get_m3u8_link(driver, output_dir)

        if m3u8_url:
            logger.info(f"🔗 Stream link: {m3u8_url[:60]}...")
            result = download_with_ffmpeg(m3u8_url, output_path, url)
            
            if result:
                return result
            else:
                logger.error("❌ Download failed")
                save_debug_info(driver, output_dir, "download_failed")
                return None
        else:
            logger.error("❌ VIDEO NOT FOUND!")
            save_debug_info(driver, output_dir, "not_found")
            return None

    except Exception as e:
        logger.error(f"❌ Critical error: {e}", exc_info=True)
        if driver:
            save_debug_info(driver, output_dir, "exception")
        return None
        
    finally:
        if driver:
            driver.quit()


# CLI interface for standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crawl anime video from URL")
    parser.add_argument("--url", required=True, help="URL to crawl")
    parser.add_argument("--anime-id", required=True, help="Anime ID")
    parser.add_argument("--episode", type=int, required=True, help="Episode number")
    parser.add_argument("--output-dir", default="./data/videos", help="Output directory")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window")
    
    args = parser.parse_args()
    
    result = crawl_episode(
        url=args.url,
        anime_id=args.anime_id,
        episode=args.episode,
        output_dir=args.output_dir,
        headless=not args.no_headless
    )
    
    if result:
        print(f"✅ Success! Video saved to: {result}")
    else:
        print("❌ Failed to crawl video")
        sys.exit(1)
