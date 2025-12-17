"""
Crawlers Package
Modular crawlers for different anime websites
"""

from .base import BaseCrawler, CrawlResult, VideoMetadata
from .vuighe import VuiGheCrawler

__all__ = [
    'BaseCrawler',
    'CrawlResult', 
    'VideoMetadata',
    'VuiGheCrawler'
]
