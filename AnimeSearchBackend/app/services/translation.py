"""
Query Refinement Service
Refines user search queries into detailed visual descriptions for CLIP-based semantic search.
Supports Gemini API (recommended), Google Translate (fallback), or Local Model.
"""

import logging
import time
from typing import Optional, Dict, Any, List
from enum import Enum
from functools import lru_cache
import threading

from app.config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# Rate Limiting Configuration for Gemini API
# ============================================================================
# Gemini free tier limits:
# - 15 requests per minute (RPM)
# - 1 million tokens per minute (TPM)
# - 1,500 requests per day (RPD)
GEMINI_MIN_INTERVAL = 4.0  # Minimum seconds between API calls (15 RPM = 4s interval)
GEMINI_TIMEOUT = 30.0      # Timeout for each API call in seconds
GEMINI_MAX_RETRIES = 3     # Maximum retry attempts
GEMINI_RETRY_DELAY = 5.0   # Initial delay between retries (exponential backoff)

# System prompt for Gemini to refine queries into visual descriptions
QUERY_REFINEMENT_SYSTEM_PROMPT = """You are an assistant that refines search queries without changing their intent. Generate **3 improved versions** of the input query that increase clarity, readability, or retrieval accuracy — but must not alter the original intent. The refined queries should be in English.
Strictly follow this format:
[Refined query 1]
[Refined query 2]
[Refined query 3]
"""


class RefinementMode(str, Enum):
    """Query refinement mode options"""
    ONLINE = "ONLINE"      # Google Translate (basic translation fallback)
    LOCAL = "LOCAL"        # HuggingFace local model (basic translation)
    GEMINI = "GEMINI"      # Google Gemini API (full refinement - recommended)


class QueryRefinementService:
    """
    Service to refine search queries into detailed visual descriptions for CLIP search.
    
    Supports 3 modes:
    - GEMINI: Google Gemini API (recommended - full query refinement with visual expansion)
    - ONLINE: Google Translate (fallback - basic translation only)
    - LOCAL: HuggingFace model (fallback - basic translation only)
    
    The GEMINI mode transforms queries like "Luffy đánh nhau" into rich descriptions
    like "Monkey D. Luffy in intense combat scene, throwing powerful punch..."
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueryRefinementService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.mode = settings.TRANSLATION_MODE.upper()
        self.translator = None
        self.model = None
        self.tokenizer = None
        
        # Cache to avoid re-processing the same queries
        self._cache: Dict[str, List[str]] = {}
        
        # Rate limiting for Gemini API
        self._last_gemini_call = 0.0
        self._gemini_lock = threading.Lock()
        
        logger.info(f"Initializing QueryRefinementService with mode: {self.mode}")
        self._setup_provider()
    
    def _setup_provider(self):
        """Initialize refinement provider based on mode"""
        try:
            if self.mode == RefinementMode.GEMINI:
                self._setup_gemini()
            elif self.mode == RefinementMode.ONLINE:
                self._setup_online()
            elif self.mode == RefinementMode.LOCAL:
                self._setup_local()
            else:
                logger.warning(f"Unknown refinement mode: {self.mode}, fallback to ONLINE")
                self.mode = RefinementMode.ONLINE
                self._setup_online()
                
            logger.info(f"Query refinement provider initialized: {self.mode}")
            
        except Exception as e:
            logger.error(f"Failed to setup refinement provider: {e}")
            logger.warning("Fallback to ONLINE mode")
            self.mode = RefinementMode.ONLINE
            self._setup_online()
    
    def _setup_gemini(self):
        """Khởi tạo Google Gemini API"""
        try:
            from google import genai
            from google.genai import types
            
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required when TRANSLATION_MODE=GEMINI. "
                    "Get your free API key at https://makersuite.google.com/app/apikey"
                )
            
            # Khởi tạo client
            self.client = genai.Client(api_key=api_key)
            
            # Model name
            model_name = settings.GEMINI_MODEL
            self.model_name = model_name
            
            logger.info(f"Gemini client initialized: {model_name}")
            
            # Test connection with a simple refinement (with rate limiting)
            try:
                test_response = self._call_gemini_with_rate_limit(
                    contents="Refine this anime search query: Luffy",
                    temperature=0.7,
                    max_output_tokens=500
                )
                logger.info(f"Gemini test successful: {test_response[:50]}")
            except Exception as test_error:
                logger.warning(f"Gemini test call failed (may be rate limited): {test_error}")
                # Don't fail initialization, just warn
            
        except ImportError:
            raise ImportError(
                "google-genai is required for GEMINI mode. "
                "Install it: pip install google-genai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    def _setup_online(self):
        """Khởi tạo Google Translate (scraped)"""
        try:
            from deep_translator import GoogleTranslator
            
            self.translator = GoogleTranslator(source='vi', target='en')
            logger.info("Google Translate (online) initialized")
            
        except ImportError:
            raise ImportError(
                "deep-translator is required for ONLINE mode. "
                "Install it: pip install deep-translator"
            )
    
    def _setup_local(self):
        """Khởi tạo HuggingFace local model"""
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            model_name = "Helsinki-NLP/opus-mt-vi-en"
            logger.info(f"Loading local translation model: {model_name}")
            
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            
            # Move to GPU if available
            if settings.DEVICE == "cuda":
                self.model = self.model.cuda()
            
            logger.info("Local translation model loaded")
            
        except ImportError:
            raise ImportError(
                "transformers is required for LOCAL mode. "
                "Install it: pip install transformers"
            )
    
    def refine(self, text: str, use_cache: bool = True) -> List[str]:
        """
        Refine a search query into multiple detailed visual descriptions for CLIP search.
        
        For GEMINI mode: Expands query with visual keywords and generates 3 variants.
        For ONLINE/LOCAL modes: Falls back to basic translation (single variant).
        
        Args:
            text: Raw user query (can be Vietnamese or English)
            use_cache: Use cache to avoid re-processing same queries
            
        Returns:
            List of refined English queries optimized for CLIP-based image search (up to 3 variants)
            
        Example:
            Input: "Luffy đánh nhau"
            Output: [
                "Monkey D. Luffy in intense combat scene, throwing powerful punch...",
                "Luffy fighting enemy, stretching rubber fist attack...",
                "Straw Hat Luffy in fierce battle, Gear technique activation..."
            ]
        """
        if not text or not text.strip():
            return [text] if text else []
        
        text = text.strip()
        
        # Check cache
        if use_cache and text in self._cache:
            logger.debug(f"Using cached refinement for: {text[:50]}")
            return self._cache[text]
        
        start_time = time.time()
        variants: List[str] = []
        
        try:
            if self.mode == RefinementMode.GEMINI:
                variants = self._refine_with_gemini(text)
            elif self.mode == RefinementMode.ONLINE:
                # Fallback: basic translation (single variant)
                translated = self._translate_online(text)
                variants = [translated] if translated else [text]
            elif self.mode == RefinementMode.LOCAL:
                # Fallback: basic translation (single variant)
                translated = self._translate_local(text)
                variants = [translated] if translated else [text]
            
            elapsed = time.time() - start_time
            logger.info(f"Query refinement completed in {elapsed:.3f}s")
            logger.info(f"  Original: '{text[:80]}'")
            logger.info(f"  Variants: {len(variants)} generated")
            for i, v in enumerate(variants):
                logger.info(f"    [{i+1}] '{v[:60]}...'" if len(v) > 60 else f"    [{i+1}] '{v}'")
            
            # Cache result
            if use_cache and variants:
                self._cache[text] = variants
            
            return variants
            
        except Exception as e:
            logger.error(f"Query refinement failed with {self.mode}: {e}")
            
            # Fallback to online translator for basic translation
            if self.mode != RefinementMode.ONLINE:
                logger.warning("Attempting fallback to ONLINE translator")
                try:
                    translated = self._translate_online_fallback(text)
                    variants = [translated] if translated else [text]
                    if use_cache:
                        self._cache[text] = variants
                    return variants
                except Exception as fallback_error:
                    logger.error(f"Fallback translation also failed: {fallback_error}")
            
            # Return original text if all methods fail
            logger.warning("All refinement methods failed, returning original text")
            return [text]
    
    # Backward compatibility alias
    def translate(self, text: str, use_cache: bool = True, target_lang: str = "en") -> str:
        """Alias for refine() - maintained for backward compatibility. Returns first variant."""
        variants = self.refine(text, use_cache)
        return variants[0] if variants else text
    
    def _refine_with_gemini(self, text: str) -> List[str]:
        """
        Refine query using Google Gemini API with visual expansion.
        
        This is the primary refinement method that transforms queries into
        detailed visual descriptions optimized for CLIP-based image search.
        Returns 3 variants of the refined query.
        
        Includes rate limiting, timeout, and retry logic to handle API limits.
        """
        if not self.client:
            raise RuntimeError("Gemini client not initialized")
        
        # Construct the prompt with system instructions
        prompt = f"""{QUERY_REFINEMENT_SYSTEM_PROMPT}

Input: "{text}"
Output:"""
        
        # Call Gemini with rate limiting and retry
        response = self._call_gemini_with_rate_limit(
            contents=prompt,
            temperature=0.8,  # Slightly higher for varied outputs
            max_output_tokens=1000
        )
        
        # Parse the response into variants (one per line)
        variants = []
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Clean up potential artifacts
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            
            # Remove any prefix like "1." or "- " or "Output:"
            prefixes_to_remove = [
                "Refined Description:",
                "Output:",
                "Result:",
                "**Refined Description:**",
                "1.", "2.", "3.",
                "- ", "* "
            ]
            for prefix in prefixes_to_remove:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
            
            if line and len(line) > 10:  # Only add meaningful variants
                variants.append(line)
        
        # Ensure we have at least 1 variant, max 3
        if not variants:
            variants = [text]  # Fallback to original
        
        return variants[:3]  # Return up to 3 variants
    
    def _call_gemini_with_rate_limit(
        self,
        contents: str,
        temperature: float = 0.7,
        max_output_tokens: int = 1000
    ) -> str:
        """
        Call Gemini API with rate limiting, timeout, and retry logic.
        
        Args:
            contents: The prompt to send to Gemini
            temperature: Generation temperature
            max_output_tokens: Maximum tokens in response
            
        Returns:
            The generated text response
            
        Raises:
            Exception if all retries fail
        """
        from google.genai import types
        import concurrent.futures
        
        last_error = None
        
        for attempt in range(GEMINI_MAX_RETRIES):
            try:
                # Rate limiting: ensure minimum interval between calls
                with self._gemini_lock:
                    elapsed = time.time() - self._last_gemini_call
                    if elapsed < GEMINI_MIN_INTERVAL:
                        wait_time = GEMINI_MIN_INTERVAL - elapsed
                        logger.debug(f"Rate limiting: waiting {wait_time:.2f}s before Gemini call")
                        time.sleep(wait_time)
                    
                    self._last_gemini_call = time.time()
                
                # Make the API call with timeout
                logger.debug(f"Gemini API call attempt {attempt + 1}/{GEMINI_MAX_RETRIES}")
                
                # Use ThreadPoolExecutor for timeout
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.client.models.generate_content,
                        model=self.model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            top_p=0.9
                        )
                    )
                    
                    try:
                        response = future.result(timeout=GEMINI_TIMEOUT)
                        return response.text.strip()
                    except concurrent.futures.TimeoutError:
                        logger.warning(f"Gemini API call timed out after {GEMINI_TIMEOUT}s")
                        raise TimeoutError(f"Gemini API call timed out after {GEMINI_TIMEOUT}s")
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                if 'resource_exhausted' in error_str or '429' in error_str or 'quota' in error_str:
                    # Extract retry delay from error message if available
                    retry_delay = GEMINI_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Gemini rate limit hit (attempt {attempt + 1}/{GEMINI_MAX_RETRIES}). "
                        f"Waiting {retry_delay:.1f}s before retry..."
                    )
                    time.sleep(retry_delay)
                    continue
                
                # For other errors, also retry with backoff
                if attempt < GEMINI_MAX_RETRIES - 1:
                    retry_delay = GEMINI_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Gemini API error (attempt {attempt + 1}/{GEMINI_MAX_RETRIES}): {e}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Gemini API failed after {GEMINI_MAX_RETRIES} attempts: {e}")
                    raise
        
        # If we get here, all retries failed
        raise last_error or RuntimeError("Gemini API call failed after all retries")
    
    def _translate_online(self, text: str) -> str:
        """Dịch bằng Google Translate (scraped)"""
        if not self.translator:
            from deep_translator import GoogleTranslator
            self.translator = GoogleTranslator(source='vi', target='en')
        
        return self.translator.translate(text)
    
    def _translate_online_fallback(self, text: str) -> str:
        """Fallback translation khi các method khác thất bại"""
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='vi', target='en')
            return translator.translate(text)
        except Exception as e:
            logger.error(f"Fallback translation failed: {e}")
            raise
    
    def _translate_local(self, text: str) -> str:
        """Dịch bằng local HuggingFace model"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Local model not initialized")
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        
        # Move to GPU if available
        if settings.DEVICE == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate translation
        outputs = self.model.generate(**inputs)
        
        # Decode
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translated
    
    def clear_cache(self):
        """Clear the refinement cache"""
        self._cache.clear()
        logger.info("Query refinement cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get refinement service statistics"""
        return {
            "mode": self.mode,
            "cache_size": len(self._cache),
            "model_info": {
                "gemini_model": settings.GEMINI_MODEL if self.mode == RefinementMode.GEMINI else None,
                "device": settings.DEVICE if self.mode == RefinementMode.LOCAL else None
            }
        }


# Singleton instance (with backward-compatible alias)
query_refinement_service = QueryRefinementService()
translation_service = query_refinement_service  # Backward compatibility alias
