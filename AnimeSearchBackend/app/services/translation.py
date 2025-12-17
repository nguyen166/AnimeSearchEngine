"""
Translation Service
Dịch thuật văn bản bằng Google Gemini, Google Translate hoặc Local Model
"""

import logging
import time
from typing import Optional, Dict, Any
from enum import Enum
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)


class TranslationMode(str, Enum):
    """Translation mode options"""
    ONLINE = "ONLINE"      # Google Translate (scraped)
    LOCAL = "LOCAL"        # HuggingFace local model
    GEMINI = "GEMINI"      # Google Gemini API


class TranslationService:
    """
    Service để dịch văn bản từ tiếng Việt sang tiếng Anh
    
    Hỗ trợ 3 chế độ:
    - ONLINE: Google Translate (miễn phí nhưng không ổn định)
    - LOCAL: HuggingFace model (chạy local, chậm hơn)
    - GEMINI: Google Gemini API (hiểu ngữ cảnh tốt, miễn phí trong quota)
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslationService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.mode = settings.TRANSLATION_MODE.upper()
        self.translator = None
        self.model = None
        self.tokenizer = None
        
        # Cache để tránh dịch lại các câu đã dịch
        self._cache: Dict[str, str] = {}
        
        logger.info(f"Initializing TranslationService with mode: {self.mode}")
        self._setup_provider()
    
    def _setup_provider(self):
        """Khởi tạo translation provider dựa trên mode"""
        try:
            if self.mode == TranslationMode.GEMINI:
                self._setup_gemini()
            elif self.mode == TranslationMode.ONLINE:
                self._setup_online()
            elif self.mode == TranslationMode.LOCAL:
                self._setup_local()
            else:
                logger.warning(f"Unknown translation mode: {self.mode}, fallback to ONLINE")
                self.mode = TranslationMode.ONLINE
                self._setup_online()
                
            logger.info(f"Translation provider initialized: {self.mode}")
            
        except Exception as e:
            logger.error(f"Failed to setup translation provider: {e}")
            logger.warning("Fallback to ONLINE mode")
            self.mode = TranslationMode.ONLINE
            self._setup_online()
    
    def _setup_gemini(self):
        """Khởi tạo Google Gemini API"""
        try:
            import google.generativeai as genai
            
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required when TRANSLATION_MODE=GEMINI. "
                    "Get your free API key at https://makersuite.google.com/app/apikey"
                )
            
            # Cấu hình Gemini
            genai.configure(api_key=api_key)
            
            # Khởi tạo model (gemini-1.5-flash miễn phí và nhanh)
            model_name = settings.GEMINI_MODEL
            self.model = genai.GenerativeModel(model_name)
            
            logger.info(f"Gemini model initialized: {model_name}")
            
            # Test connection
            test_response = self.model.generate_content(
                "Translate to English: Xin chào",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=100
                )
            )
            logger.info(f"Gemini test successful: {test_response.text[:50]}")
            
        except ImportError:
            raise ImportError(
                "google-generativeai is required for GEMINI mode. "
                "Install it: pip install google-generativeai"
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
    
    def translate(self, text: str, use_cache: bool = True) -> str:
        """
        Dịch văn bản từ tiếng Việt sang tiếng Anh
        
        Args:
            text: Văn bản cần dịch
            use_cache: Sử dụng cache để tránh dịch lại
            
        Returns:
            Văn bản đã dịch
        """
        if not text or not text.strip():
            return text
        
        # Check cache
        if use_cache and text in self._cache:
            logger.debug(f"Using cached translation for: {text[:50]}")
            return self._cache[text]
        
        start_time = time.time()
        translated = ""
        
        try:
            if self.mode == TranslationMode.GEMINI:
                translated = self._translate_gemini(text)
            elif self.mode == TranslationMode.ONLINE:
                translated = self._translate_online(text)
            elif self.mode == TranslationMode.LOCAL:
                translated = self._translate_local(text)
            
            elapsed = time.time() - start_time
            logger.info(f"Translation completed in {elapsed:.3f}s: '{text[:50]}' -> '{translated[:50]}'")
            
            # Cache result
            if use_cache:
                self._cache[text] = translated
            
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed with {self.mode}: {e}")
            
            # Fallback to online translator
            if self.mode != TranslationMode.ONLINE:
                logger.warning("Attempting fallback to ONLINE translator")
                try:
                    translated = self._translate_online_fallback(text)
                    if use_cache:
                        self._cache[text] = translated
                    return translated
                except Exception as fallback_error:
                    logger.error(f"Fallback translation also failed: {fallback_error}")
            
            # Return original text if all fails
            logger.warning("All translation methods failed, returning original text")
            return text
    
    def _translate_gemini(self, text: str) -> str:
        """Dịch bằng Google Gemini API"""
        if not self.model:
            raise RuntimeError("Gemini model not initialized")
        
        # Prompt engineering để đảm bảo output clean
        prompt = f"""Translate the following Vietnamese text to English. 
Return ONLY the translated text, no explanation or additional words.
Preserve anime terminology (e.g., "Haki", "Bankai", "Chakra") when appropriate.

Vietnamese text: {text}

English translation:"""
        
        try:
            import google.generativeai as genai
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent translation
                    max_output_tokens=500,
                    top_p=0.8
                )
            )
            
            # Extract text from response
            translated = response.text.strip()
            
            # Clean up potential artifacts
            if translated.startswith('"') and translated.endswith('"'):
                translated = translated[1:-1]
            
            return translated
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
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
        """Xóa cache translation"""
        self._cache.clear()
        logger.info("Translation cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê translation service"""
        return {
            "mode": self.mode,
            "cache_size": len(self._cache),
            "model_info": {
                "gemini_model": settings.GEMINI_MODEL if self.mode == TranslationMode.GEMINI else None,
                "device": settings.DEVICE if self.mode == TranslationMode.LOCAL else None
            }
        }


# Singleton instance
translation_service = TranslationService()
