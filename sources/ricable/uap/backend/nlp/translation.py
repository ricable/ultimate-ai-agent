# File: backend/nlp/translation.py
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from ..monitoring.logs.logger import uap_logger, LogLevel, EventType
from ..cache.decorators import cache_translation

@dataclass
class TranslationResult:
    """Translation result"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    method: str  # rule_based, statistical, neural
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "confidence": self.confidence,
            "method": self.method
        }

@dataclass
class LanguageDetectionResult:
    """Language detection result"""
    language: str
    confidence: float
    alternatives: List[Tuple[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "confidence": self.confidence,
            "alternatives": [{"language": lang, "confidence": conf} for lang, conf in self.alternatives]
        }

class TranslationService:
    """
    Advanced Translation and Multi-Language Support
    
    Provides comprehensive language processing including:
    - Language detection
    - Text translation between multiple languages
    - Language-specific text processing
    - Multi-language content management
    - Translation quality assessment
    """
    
    def __init__(self):
        self._initialized = False
        
        # Language detection patterns
        self.language_patterns = {
            "en": {
                "common_words": {
                    "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
                    "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                    "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can",
                    "this", "that", "these", "those", "there", "here", "where", "when", "what", "how", "why"
                },
                "patterns": [
                    r"\bthe\b", r"\band\b", r"\bof\b", r"\bto\b", r"\ba\b", r"\bin\b", r"\bis\b"
                ],
                "name": "English"
            },
            "es": {
                "common_words": {
                    "el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo", "le",
                    "da", "su", "por", "son", "con", "para", "al", "del", "los", "las", "una", "está",
                    "como", "pero", "más", "todo", "muy", "bien", "ya", "vez", "si", "ser", "hacer"
                },
                "patterns": [
                    r"\bel\b", r"\bla\b", r"\bde\b", r"\bque\b", r"\by\b", r"\ben\b", r"\bun\b"
                ],
                "name": "Spanish"
            },
            "fr": {
                "common_words": {
                    "le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir", "que", "pour",
                    "dans", "ce", "son", "une", "sur", "avec", "ne", "se", "pas", "tout", "plus",
                    "par", "grand", "en", "me", "même", "y", "ces", "suite", "parce", "elle", "vous"
                },
                "patterns": [
                    r"\ble\b", r"\bde\b", r"\bet\b", r"\bà\b", r"\bun\b", r"\bil\b", r"\ben\b"
                ],
                "name": "French"
            },
            "de": {
                "common_words": {
                    "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des", "auf",
                    "für", "ist", "im", "dem", "nicht", "ein", "eine", "als", "auch", "es", "an", "werden",
                    "aus", "er", "hat", "dass", "sie", "nach", "wird", "bei", "einer", "um", "am", "sind"
                },
                "patterns": [
                    r"\bder\b", r"\bdie\b", r"\bund\b", r"\bin\b", r"\bden\b", r"\bvon\b", r"\bzu\b"
                ],
                "name": "German"
            },
            "it": {
                "common_words": {
                    "il", "di", "che", "e", "la", "il", "un", "a", "per", "non", "in", "una", "sono",
                    "mi", "ho", "lo", "ma", "se", "come", "da", "quando", "anche", "ci", "questo",
                    "qui", "lei", "o", "me", "cosa", "tempo", "molto", "fare", "suo", "lei", "mio"
                },
                "patterns": [
                    r"\bil\b", r"\bdi\b", r"\bche\b", r"\be\b", r"\bla\b", r"\bun\b", r"\ba\b"
                ],
                "name": "Italian"
            },
            "pt": {
                "common_words": {
                    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não",
                    "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi",
                    "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito", "há"
                },
                "patterns": [
                    r"\bde\b", r"\ba\b", r"\bo\b", r"\bque\b", r"\be\b", r"\bdo\b", r"\bda\b"
                ],
                "name": "Portuguese"
            },
            "ru": {
                "common_words": {
                    "в", "и", "не", "на", "я", "быть", "то", "он", "с", "а", "как", "это", "по",
                    "но", "они", "мы", "что", "за", "из", "так", "же", "от", "при", "до", "для"
                },
                "patterns": [
                    r"\bв\b", r"\bи\b", r"\bне\b", r"\bна\b", r"\bя\b", r"\bто\b", r"\bон\b"
                ],
                "name": "Russian"
            },
            "zh": {
                "common_words": {
                    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "个",
                    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没", "看", "好"
                },
                "patterns": [
                    r"的", r"了", r"在", r"是", r"我", r"有", r"和"
                ],
                "name": "Chinese"
            },
            "ja": {
                "common_words": {
                    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "ある",
                    "いる", "も", "する", "から", "な", "こと", "として", "い", "や", "れる", "など"
                },
                "patterns": [
                    r"の", r"に", r"は", r"を", r"た", r"が", r"で"
                ],
                "name": "Japanese"
            },
            "ar": {
                "common_words": {
                    "في", "من", "إلى", "على", "أن", "هذا", "هذه", "التي", "الذي", "أو", "كان",
                    "كل", "عند", "عن", "مع", "قد", "لا", "ما", "هو", "هي", "ذلك", "تلك", "بعد"
                },
                "patterns": [
                    r"في", r"من", r"إلى", r"على", r"أن", r"هذا", r"هذه"
                ],
                "name": "Arabic"
            },
            "hi": {
                "common_words": {
                    "और", "का", "एक", "में", "की", "है", "यह", "को", "से", "कि", "जो", "कर",
                    "पर", "गया", "था", "हैं", "अपने", "तो", "ही", "या", "हो", "भी", "कह", "दिया"
                },
                "patterns": [
                    r"और", r"का", r"एक", r"में", r"की", r"है", r"यह"
                ],
                "name": "Hindi"
            }
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        
        # Basic translation dictionaries (for demonstration)
        self.basic_translations = {
            ("en", "es"): {
                "hello": "hola",
                "goodbye": "adiós",
                "please": "por favor",
                "thank you": "gracias",
                "yes": "sí",
                "no": "no",
                "help": "ayuda",
                "error": "error",
                "success": "éxito",
                "welcome": "bienvenido"
            },
            ("en", "fr"): {
                "hello": "bonjour",
                "goodbye": "au revoir",
                "please": "s'il vous plaît",
                "thank you": "merci",
                "yes": "oui",
                "no": "non",
                "help": "aide",
                "error": "erreur",
                "success": "succès",
                "welcome": "bienvenue"
            },
            ("en", "de"): {
                "hello": "hallo",
                "goodbye": "auf wiedersehen",
                "please": "bitte",
                "thank you": "danke",
                "yes": "ja",
                "no": "nein",
                "help": "hilfe",
                "error": "fehler",
                "success": "erfolg",
                "welcome": "willkommen"
            },
            ("en", "it"): {
                "hello": "ciao",
                "goodbye": "arrivederci",
                "please": "per favore",
                "thank you": "grazie",
                "yes": "sì",
                "no": "no",
                "help": "aiuto",
                "error": "errore",
                "success": "successo",
                "welcome": "benvenuto"
            }
        }
        
        # Language metadata
        self.language_info = {
            "en": {"name": "English", "native_name": "English", "direction": "ltr", "family": "Germanic"},
            "es": {"name": "Spanish", "native_name": "Español", "direction": "ltr", "family": "Romance"},
            "fr": {"name": "French", "native_name": "Français", "direction": "ltr", "family": "Romance"},
            "de": {"name": "German", "native_name": "Deutsch", "direction": "ltr", "family": "Germanic"},
            "it": {"name": "Italian", "native_name": "Italiano", "direction": "ltr", "family": "Romance"},
            "pt": {"name": "Portuguese", "native_name": "Português", "direction": "ltr", "family": "Romance"},
            "ru": {"name": "Russian", "native_name": "Русский", "direction": "ltr", "family": "Slavic"},
            "zh": {"name": "Chinese", "native_name": "中文", "direction": "ltr", "family": "Sino-Tibetan"},
            "ja": {"name": "Japanese", "native_name": "日本語", "direction": "ltr", "family": "Japonic"},
            "ar": {"name": "Arabic", "native_name": "العربية", "direction": "rtl", "family": "Semitic"},
            "hi": {"name": "Hindi", "native_name": "हिन्दी", "direction": "ltr", "family": "Indo-European"}
        }
        
        # Statistics
        self.translation_stats = {
            "total_translations": 0,
            "successful_translations": 0,
            "failed_translations": 0,
            "language_detections": 0,
            "language_distribution": {},
            "translation_pairs": {},
            "avg_processing_time": 0
        }
    
    async def initialize(self):
        """Initialize translation service"""
        if self._initialized:
            return
        
        try:
            # Compile language detection patterns
            for lang_code, config in self.language_patterns.items():
                patterns = config.get("patterns", [])
                if patterns:
                    combined_pattern = "|".join(patterns)
                    self.compiled_patterns[lang_code] = re.compile(combined_pattern, re.IGNORECASE | re.UNICODE)
            
            self._initialized = True
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Translation Service initialized successfully",
                EventType.SYSTEM,
                {
                    "supported_languages": len(self.language_patterns),
                    "translation_pairs": len(self.basic_translations)
                },
                "translation_service"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Translation Service initialization failed: {str(e)}",
                EventType.SYSTEM,
                {"error": str(e)},
                "translation_service"
            )
            raise
    
    async def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language code
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            detection_result = await self.detect_language_detailed(text)
            
            self.translation_stats["language_detections"] += 1
            detected_lang = detection_result.language
            
            if detected_lang not in self.translation_stats["language_distribution"]:
                self.translation_stats["language_distribution"][detected_lang] = 0
            self.translation_stats["language_distribution"][detected_lang] += 1
            
            return detected_lang
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Language detection failed: {str(e)}",
                EventType.NLP,
                {"error": str(e), "text_length": len(text)},
                "translation_service"
            )
            return "en"  # Default to English
    
    async def detect_language_detailed(self, text: str) -> LanguageDetectionResult:
        """
        Detect language with detailed results
        
        Args:
            text: Text to analyze
            
        Returns:
            LanguageDetectionResult with detailed information
        """
        if not self._initialized:
            await self.initialize()
        
        if not text or len(text.strip()) == 0:
            return LanguageDetectionResult("en", 0.0, [])
        
        # Preprocess text
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return LanguageDetectionResult("en", 0.0, [])
        
        # Calculate scores for each language
        language_scores = {}
        
        for lang_code, config in self.language_patterns.items():
            score = self._calculate_language_score(text_lower, words, lang_code, config)
            if score > 0:
                language_scores[lang_code] = score
        
        if not language_scores:
            return LanguageDetectionResult("en", 0.0, [])
        
        # Sort by score
        sorted_languages = sorted(language_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get best match and alternatives
        best_language = sorted_languages[0][0]
        best_score = sorted_languages[0][1]
        
        # Normalize confidence
        total_score = sum(language_scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.0
        
        # Get top 3 alternatives
        alternatives = [(lang, score / total_score) for lang, score in sorted_languages[1:4]]
        
        uap_logger.log_event(
            LogLevel.INFO,
            f"Language detected: {best_language} ({confidence:.3f})",
            EventType.NLP,
            {
                "detected_language": best_language,
                "confidence": confidence,
                "text_length": len(text),
                "alternatives": len(alternatives)
            },
            "translation_service"
        )
        
        return LanguageDetectionResult(best_language, confidence, alternatives)
    
    def _calculate_language_score(
        self, 
        text: str, 
        words: List[str], 
        lang_code: str,
        config: Dict[str, Any]
    ) -> float:
        """Calculate language detection score"""
        score = 0.0
        
        # Check common words
        common_words = config.get("common_words", set())
        if common_words:
            word_matches = len(set(words).intersection(common_words))
            if word_matches > 0:
                # Weight by word frequency
                word_score = word_matches / len(words)
                score += word_score * 0.7
        
        # Check patterns
        if lang_code in self.compiled_patterns:
            pattern = self.compiled_patterns[lang_code]
            matches = pattern.findall(text)
            if matches:
                pattern_score = min(1.0, len(matches) / 10)  # Normalize
                score += pattern_score * 0.3
        
        # Language-specific adjustments
        if lang_code in ["zh", "ja", "ar", "hi"]:
            # For non-Latin scripts, check for specific character ranges
            if lang_code == "zh":
                chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
                if chinese_chars > 0:
                    score += min(1.0, chinese_chars / len(text)) * 0.8
            
            elif lang_code == "ja":
                japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text))
                if japanese_chars > 0:
                    score += min(1.0, japanese_chars / len(text)) * 0.8
            
            elif lang_code == "ar":
                arabic_chars = len(re.findall(r'[\u0600-\u06ff]', text))
                if arabic_chars > 0:
                    score += min(1.0, arabic_chars / len(text)) * 0.8
            
            elif lang_code == "hi":
                hindi_chars = len(re.findall(r'[\u0900-\u097f]', text))
                if hindi_chars > 0:
                    score += min(1.0, hindi_chars / len(text)) * 0.8
        
        return score
    
    @cache_translation
    async def translate_text(
        self, 
        text: str, 
        target_language: str,
        source_language: Optional[str] = None
    ) -> TranslationResult:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code (auto-detect if None)
            
        Returns:
            TranslationResult with translation information
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Detect source language if not provided
            if source_language is None:
                source_language = await self.detect_language(text)
            
            # Check if translation is needed
            if source_language == target_language:
                result = TranslationResult(
                    original_text=text,
                    translated_text=text,
                    source_language=source_language,
                    target_language=target_language,
                    confidence=1.0,
                    method="no_translation_needed"
                )
            else:
                # Perform translation
                result = await self._translate_text_internal(text, source_language, target_language)
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Update statistics
            self.translation_stats["total_translations"] += 1
            self.translation_stats["successful_translations"] += 1
            
            pair_key = f"{source_language}→{target_language}"
            if pair_key not in self.translation_stats["translation_pairs"]:
                self.translation_stats["translation_pairs"][pair_key] = 0
            self.translation_stats["translation_pairs"][pair_key] += 1
            
            self.translation_stats["avg_processing_time"] = (
                (self.translation_stats["avg_processing_time"] * (self.translation_stats["total_translations"] - 1) + processing_time) /
                self.translation_stats["total_translations"]
            )
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Translation completed: {source_language}→{target_language} ({result.confidence:.3f})",
                EventType.NLP,
                {
                    "source_language": source_language,
                    "target_language": target_language,
                    "confidence": result.confidence,
                    "method": result.method,
                    "text_length": len(text),
                    "processing_time_ms": processing_time
                },
                "translation_service"
            )
            
            return result
            
        except Exception as e:
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self.translation_stats["failed_translations"] += 1
            
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Translation failed: {str(e)}",
                EventType.NLP,
                {
                    "source_language": source_language,
                    "target_language": target_language,
                    "error": str(e),
                    "text_length": len(text),
                    "processing_time_ms": processing_time
                },
                "translation_service"
            )
            
            raise
    
    async def _translate_text_internal(
        self, 
        text: str, 
        source_language: str,
        target_language: str
    ) -> TranslationResult:
        """Internal translation logic"""
        
        # Check for basic dictionary translation
        translation_key = (source_language, target_language)
        
        if translation_key in self.basic_translations:
            translated_text = self._apply_dictionary_translation(
                text, self.basic_translations[translation_key]
            )
            confidence = 0.7  # Basic translation confidence
            method = "rule_based"
        else:
            # Fallback: return original text with note
            translated_text = f"[Translation from {source_language} to {target_language} not available] {text}"
            confidence = 0.3
            method = "fallback"
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            confidence=confidence,
            method=method
        )
    
    def _apply_dictionary_translation(self, text: str, translation_dict: Dict[str, str]) -> str:
        """Apply dictionary-based translation"""
        
        # Simple word-by-word translation
        words = re.findall(r'\b\w+\b|\W+', text.lower())
        translated_words = []
        
        for word in words:
            if word.strip() and word.strip() in translation_dict:
                translated_words.append(translation_dict[word.strip()])
            else:
                translated_words.append(word)
        
        return ''.join(translated_words)
    
    async def translate_conversation(
        self, 
        messages: List[Dict[str, Any]], 
        target_language: str
    ) -> List[Dict[str, Any]]:
        """
        Translate entire conversation to target language
        
        Args:
            messages: List of conversation messages
            target_language: Target language code
            
        Returns:
            List of translated messages
        """
        try:
            translated_messages = []
            
            for message in messages:
                if "content" in message and message["content"]:
                    # Translate message content
                    translation_result = await self.translate_text(
                        message["content"], 
                        target_language
                    )
                    
                    # Create translated message
                    translated_message = message.copy()
                    translated_message["content"] = translation_result.translated_text
                    translated_message["translation"] = {
                        "original_content": translation_result.original_text,
                        "source_language": translation_result.source_language,
                        "target_language": translation_result.target_language,
                        "confidence": translation_result.confidence,
                        "method": translation_result.method
                    }
                    
                    translated_messages.append(translated_message)
                else:
                    # Keep non-text messages as-is
                    translated_messages.append(message)
            
            return translated_messages
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Conversation translation failed: {str(e)}",
                EventType.NLP,
                {"error": str(e), "target_language": target_language},
                "translation_service"
            )
            raise
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """Get list of supported languages"""
        return [
            {
                "code": code,
                "name": info["name"],
                "native_name": info["native_name"],
                "direction": info["direction"],
                "family": info["family"]
            }
            for code, info in self.language_info.items()
        ]
    
    def get_language_info(self, language_code: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a language"""
        if language_code in self.language_info:
            return self.language_info[language_code].copy()
        return None
    
    def is_rtl_language(self, language_code: str) -> bool:
        """Check if language is right-to-left"""
        info = self.get_language_info(language_code)
        return info and info.get("direction") == "rtl"
    
    async def batch_translate(
        self, 
        texts: List[str], 
        target_language: str,
        source_language: Optional[str] = None
    ) -> List[TranslationResult]:
        """
        Translate multiple texts in batch
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code (auto-detect if None)
            
        Returns:
            List of TranslationResult objects
        """
        try:
            # Process translations in parallel
            translation_tasks = [
                self.translate_text(text, target_language, source_language)
                for text in texts
            ]
            
            results = await asyncio.gather(*translation_tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in results if isinstance(r, TranslationResult)]
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Batch translation completed: {len(valid_results)}/{len(texts)} successful",
                EventType.NLP,
                {
                    "total_texts": len(texts),
                    "successful_translations": len(valid_results),
                    "target_language": target_language
                },
                "translation_service"
            )
            
            return valid_results
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Batch translation failed: {str(e)}",
                EventType.NLP,
                {"error": str(e), "batch_size": len(texts)},
                "translation_service"
            )
            raise
    
    async def get_translation_quality_score(
        self, 
        original_text: str, 
        translated_text: str,
        source_language: str,
        target_language: str
    ) -> float:
        """
        Assess translation quality (basic implementation)
        
        Args:
            original_text: Original text
            translated_text: Translated text
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Quality score from 0.0 to 1.0
        """
        try:
            # Basic quality assessment
            quality_score = 0.5  # Base score
            
            # Check if translation actually changed
            if original_text.lower() == translated_text.lower():
                if source_language == target_language:
                    quality_score = 1.0  # Perfect for same language
                else:
                    quality_score = 0.2  # Poor if should have changed
            
            # Check length ratio (reasonable translations shouldn't be drastically different in length)
            length_ratio = len(translated_text) / len(original_text) if original_text else 1.0
            
            if 0.5 <= length_ratio <= 2.0:
                quality_score += 0.2
            else:
                quality_score -= 0.1
            
            # Check for preserved entities (URLs, emails, numbers)
            original_entities = re.findall(r'\b\w+@\w+\.\w+\b|https?://\S+|\b\d+\b', original_text)
            translated_entities = re.findall(r'\b\w+@\w+\.\w+\b|https?://\S+|\b\d+\b', translated_text)
            
            entity_preservation_ratio = len(translated_entities) / len(original_entities) if original_entities else 1.0
            quality_score += min(0.2, entity_preservation_ratio * 0.2)
            
            # Ensure score is in valid range
            quality_score = max(0.0, min(1.0, quality_score))
            
            return quality_score
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.WARNING,
                f"Translation quality assessment failed: {str(e)}",
                EventType.NLP,
                {"error": str(e)},
                "translation_service"
            )
            return 0.5  # Default neutral score
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get translation service statistics"""
        return {
            **self.translation_stats,
            "supported_languages": list(self.language_info.keys()),
            "translation_pairs_available": list(self.basic_translations.keys()),
            "initialized": self._initialized
        }
    
    async def cleanup(self):
        """Clean up translation service resources"""
        try:
            self.compiled_patterns.clear()
            self.language_patterns.clear()
            self.basic_translations.clear()
            self.language_info.clear()
            
            self._initialized = False
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Translation Service cleanup completed",
                EventType.SYSTEM,
                self.translation_stats,
                "translation_service"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Translation Service cleanup failed: {str(e)}",
                EventType.SYSTEM,
                {"error": str(e)},
                "translation_service"
            )