"""
Enhanced OCR with Layout Understanding Module

Provides advanced OCR capabilities including:
- Multi-engine OCR (Tesseract, EasyOCR, PaddleOCR)
- Document layout analysis and understanding
- Table detection and extraction
- Reading order detection
- Text region classification
- Language detection and multi-language support
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
import json
import re

try:
    import cv2
    from PIL import Image
    import pytesseract
    import easyocr
    from paddleocr import PaddleOCR
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    import layoutparser as lp
    from sklearn.cluster import DBSCAN
    HAS_OCR_DEPS = True
except ImportError:
    HAS_OCR_DEPS = False
    cv2 = None
    Image = None
    pytesseract = None
    easyocr = None
    PaddleOCR = None

logger = logging.getLogger(__name__)

class EnhancedOCR:
    """Enhanced OCR with layout understanding and multi-engine support."""
    
    def __init__(self, model_cache_dir: Optional[str] = None):
        """Initialize enhanced OCR with multiple engines."""
        self.model_cache_dir = model_cache_dir or "./models/ocr"
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.engines = {}
        self.layout_models = {}
        
        if not HAS_OCR_DEPS:
            logger.warning("OCR dependencies not installed. Limited functionality available.")
            return
            
        # Initialize OCR engines
        self._initialize_engines()
        self._initialize_layout_models()
    
    def _initialize_engines(self):
        """Initialize OCR engines."""
        try:
            # EasyOCR - supports many languages
            self.engines['easyocr'] = easyocr.Reader(['en', 'es', 'fr', 'de', 'it', 'pt', 'zh'])
            logger.info("EasyOCR initialized")
            
            # PaddleOCR - good for Asian languages
            self.engines['paddleocr'] = PaddleOCR(use_angle_cls=True, lang='en')
            logger.info("PaddleOCR initialized")
            
            # Tesseract configuration
            self.tesseract_config = '--oem 3 --psm 6'
            logger.info("Tesseract configured")
            
        except Exception as e:
            logger.error(f"Error initializing OCR engines: {e}")
    
    def _initialize_layout_models(self):
        """Initialize layout analysis models."""
        try:
            # LayoutParser models for document layout analysis
            self.layout_models['newspaper'] = lp.AutoLayoutModel(
                'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
            )
            
            # Table detection model
            self.layout_models['table'] = lp.AutoLayoutModel(
                'lp://TableBank/faster_rcnn_R_50_FPN_3x/config'
            )
            
            logger.info("Layout models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing layout models: {e}")
            self.layout_models = {}
    
    async def extract_text(self, image_data: Union[str, bytes, np.ndarray], 
                          engines: List[str] = None, 
                          enhance_image: bool = True,
                          detect_language: bool = True) -> Dict[str, Any]:
        """
        Extract text from image using multiple OCR engines.
        
        Args:
            image_data: Image data in various formats
            engines: List of engines to use ['tesseract', 'easyocr', 'paddleocr']
            enhance_image: Whether to enhance image before OCR
            detect_language: Whether to detect language
        
        Returns:
            Dictionary with OCR results from all engines
        """
        if not HAS_OCR_DEPS:
            return {"error": "OCR dependencies not installed"}
        
        try:
            # Load and preprocess image
            image = await self._load_image(image_data)
            if image is None:
                return {"error": "Failed to load image"}
            
            if enhance_image:
                image = await self._enhance_for_ocr(image)
            
            # Default engines if not specified
            if engines is None:
                engines = ['tesseract', 'easyocr', 'paddleocr']
            
            results = {
                "image_size": image.shape[:2],
                "engines_used": engines,
                "results": {}
            }
            
            # Run OCR with different engines
            for engine in engines:
                if engine in ['tesseract', 'easyocr', 'paddleocr']:
                    engine_result = await self._run_ocr_engine(image, engine)
                    results["results"][engine] = engine_result
            
            # Combine results and find consensus
            combined_text = await self._combine_ocr_results(results["results"])
            results["combined_text"] = combined_text
            
            # Language detection
            if detect_language:
                results["language"] = await self._detect_language(combined_text["text"])
            
            # Confidence scoring
            results["confidence_score"] = await self._calculate_confidence(results["results"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in OCR text extraction: {e}")
            return {"error": str(e)}
    
    async def analyze_layout(self, image_data: Union[str, bytes, np.ndarray], 
                           include_reading_order: bool = True) -> Dict[str, Any]:
        """
        Analyze document layout and structure.
        
        Args:
            image_data: Image data
            include_reading_order: Whether to determine reading order
        
        Returns:
            Dictionary with layout analysis results
        """
        if not HAS_OCR_DEPS or not self.layout_models:
            return {"error": "Layout analysis dependencies not available"}
        
        try:
            # Load image
            image = await self._load_image(image_data)
            if image is None:
                return {"error": "Failed to load image"}
            
            results = {
                "image_size": image.shape[:2],
                "layout_elements": [],
                "statistics": {}
            }
            
            # Detect layout elements
            if 'newspaper' in self.layout_models:
                layout = self.layout_models['newspaper'].detect(image)
                
                for element in layout:
                    element_info = {
                        "type": element.type,
                        "bbox": [element.x_1, element.y_1, element.x_2, element.y_2],
                        "confidence": float(element.score),
                        "area": (element.x_2 - element.x_1) * (element.y_2 - element.y_1)
                    }
                    results["layout_elements"].append(element_info)
            
            # Calculate statistics
            element_types = [elem["type"] for elem in results["layout_elements"]]
            results["statistics"] = {
                "total_elements": len(results["layout_elements"]),
                "element_counts": {elem_type: element_types.count(elem_type) 
                                for elem_type in set(element_types)},
                "avg_confidence": np.mean([elem["confidence"] 
                                        for elem in results["layout_elements"]]) if results["layout_elements"] else 0
            }
            
            # Reading order detection
            if include_reading_order and results["layout_elements"]:
                results["reading_order"] = await self._determine_reading_order(results["layout_elements"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in layout analysis: {e}")
            return {"error": str(e)}
    
    async def extract_tables(self, image_data: Union[str, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Extract tables from document image.
        
        Args:
            image_data: Image data
        
        Returns:
            Dictionary with extracted table data
        """
        if not HAS_OCR_DEPS:
            return {"error": "Table extraction dependencies not available"}
        
        try:
            # Load image
            image = await self._load_image(image_data)
            if image is None:
                return {"error": "Failed to load image"}
            
            results = {
                "image_size": image.shape[:2],
                "tables": []
            }
            
            # Detect table regions
            table_regions = await self._detect_table_regions(image)
            
            for i, region in enumerate(table_regions):
                # Extract table region
                x1, y1, x2, y2 = region["bbox"]
                table_img = image[y1:y2, x1:x2]
                
                # Extract table structure and content
                table_data = await self._extract_table_structure(table_img)
                
                table_info = {
                    "table_id": i,
                    "bbox": region["bbox"],
                    "confidence": region["confidence"],
                    "rows": table_data["rows"],
                    "cols": table_data["cols"],
                    "cells": table_data["cells"],
                    "data": table_data["data"]
                }
                
                results["tables"].append(table_info)
            
            results["tables_found"] = len(results["tables"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in table extraction: {e}")
            return {"error": str(e)}
    
    async def _load_image(self, image_data: Union[str, bytes, np.ndarray]) -> Optional[np.ndarray]:
        """Load image from various input formats."""
        try:
            if isinstance(image_data, str):
                # Base64 string or file path
                if image_data.startswith('data:image'):
                    import base64
                    image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    return np.array(image)
                else:
                    # File path
                    image = cv2.imread(image_data)
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
            elif isinstance(image_data, bytes):
                import io
                image = Image.open(io.BytesIO(image_data))
                return np.array(image)
                
            elif isinstance(image_data, np.ndarray):
                return image_data
                
            else:
                logger.error(f"Unsupported image data type: {type(image_data)}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    async def _enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better OCR results."""
        try:
            # Convert to grayscale if color
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Threshold for binary image
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return binary
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image
    
    async def _run_ocr_engine(self, image: np.ndarray, engine: str) -> Dict[str, Any]:
        """Run specific OCR engine on image."""
        try:
            if engine == 'tesseract':
                return await self._run_tesseract(image)
            elif engine == 'easyocr':
                return await self._run_easyocr(image)
            elif engine == 'paddleocr':
                return await self._run_paddleocr(image)
            else:
                return {"error": f"Unknown OCR engine: {engine}"}
                
        except Exception as e:
            logger.error(f"Error running {engine}: {e}")
            return {"error": str(e)}
    
    async def _run_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """Run Tesseract OCR."""
        try:
            # Extract text
            text = pytesseract.image_to_string(image, config=self.tesseract_config)
            
            # Extract detailed data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Process word-level data
            words = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:
                    word_info = {
                        "text": data['text'][i],
                        "confidence": int(data['conf'][i]),
                        "bbox": [
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ]
                    }
                    words.append(word_info)
            
            return {
                "text": text.strip(),
                "words": words,
                "avg_confidence": np.mean([w["confidence"] for w in words]) if words else 0,
                "engine": "tesseract"
            }
            
        except Exception as e:
            return {"error": f"Tesseract failed: {e}"}
    
    async def _run_easyocr(self, image: np.ndarray) -> Dict[str, Any]:
        """Run EasyOCR."""
        try:
            if 'easyocr' not in self.engines:
                return {"error": "EasyOCR not initialized"}
            
            results = self.engines['easyocr'].readtext(image)
            
            words = []
            all_text = []
            
            for (bbox, text, confidence) in results:
                # Convert bbox format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                bbox_rect = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                
                word_info = {
                    "text": text,
                    "confidence": float(confidence * 100),  # Convert to percentage
                    "bbox": bbox_rect
                }
                words.append(word_info)
                all_text.append(text)
            
            return {
                "text": " ".join(all_text),
                "words": words,
                "avg_confidence": np.mean([w["confidence"] for w in words]) if words else 0,
                "engine": "easyocr"
            }
            
        except Exception as e:
            return {"error": f"EasyOCR failed: {e}"}
    
    async def _run_paddleocr(self, image: np.ndarray) -> Dict[str, Any]:
        """Run PaddleOCR."""
        try:
            if 'paddleocr' not in self.engines:
                return {"error": "PaddleOCR not initialized"}
            
            results = self.engines['paddleocr'].ocr(image, cls=True)
            
            words = []
            all_text = []
            
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, confidence) = line
                    
                    # Convert bbox format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    bbox_rect = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    
                    word_info = {
                        "text": text,
                        "confidence": float(confidence * 100),  # Convert to percentage
                        "bbox": bbox_rect
                    }
                    words.append(word_info)
                    all_text.append(text)
            
            return {
                "text": " ".join(all_text),
                "words": words,
                "avg_confidence": np.mean([w["confidence"] for w in words]) if words else 0,
                "engine": "paddleocr"
            }
            
        except Exception as e:
            return {"error": f"PaddleOCR failed: {e}"}
    
    async def _combine_ocr_results(self, engine_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine results from multiple OCR engines."""
        try:
            # Collect all text and confidence scores
            texts = []
            confidences = []
            all_words = []
            
            for engine, result in engine_results.items():
                if "error" not in result:
                    texts.append(result.get("text", ""))
                    confidences.append(result.get("avg_confidence", 0))
                    all_words.extend(result.get("words", []))
            
            if not texts:
                return {"text": "", "confidence": 0, "word_count": 0}
            
            # Simple consensus: use text from engine with highest confidence
            best_idx = np.argmax(confidences) if confidences else 0
            best_text = texts[best_idx] if texts else ""
            
            # Word-level consensus using clustering
            consensus_words = await self._get_word_consensus(all_words)
            
            return {
                "text": best_text,
                "confidence": max(confidences) if confidences else 0,
                "word_count": len(best_text.split()),
                "consensus_words": consensus_words,
                "engines_agreed": len([t for t in texts if t == best_text])
            }
            
        except Exception as e:
            logger.error(f"Error combining OCR results: {e}")
            return {"text": "", "confidence": 0, "word_count": 0}
    
    async def _get_word_consensus(self, all_words: List[Dict]) -> List[Dict]:
        """Get consensus on word-level results from multiple engines."""
        try:
            if not all_words:
                return []
            
            # Group words by spatial proximity
            positions = np.array([[w["bbox"][0], w["bbox"][1]] for w in all_words])
            
            if len(positions) > 1:
                # Use DBSCAN clustering to group nearby words
                clustering = DBSCAN(eps=50, min_samples=1).fit(positions)
                labels = clustering.labels_
                
                consensus_words = []
                for cluster_id in set(labels):
                    if cluster_id == -1:  # Noise points
                        continue
                    
                    # Get words in this cluster
                    cluster_words = [all_words[i] for i, label in enumerate(labels) if label == cluster_id]
                    
                    # Find best word in cluster (highest confidence)
                    best_word = max(cluster_words, key=lambda w: w["confidence"])
                    consensus_words.append(best_word)
                
                return consensus_words
            else:
                return all_words
                
        except Exception as e:
            logger.error(f"Error in word consensus: {e}")
            return all_words
    
    async def _detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of extracted text."""
        try:
            if not text.strip():
                return {"language": "unknown", "confidence": 0}
            
            # Simple language detection based on character patterns
            # This is a basic implementation - could be enhanced with langdetect library
            
            # Count character types
            latin_chars = len(re.findall(r'[a-zA-Z]', text))
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
            arabic_chars = len(re.findall(r'[\u0600-\u06ff]', text))
            
            total_chars = len(re.findall(r'[a-zA-Z\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u0600-\u06ff]', text))
            
            if total_chars == 0:
                return {"language": "unknown", "confidence": 0}
            
            # Determine dominant language
            ratios = {
                "english": latin_chars / total_chars,
                "chinese": chinese_chars / total_chars,
                "japanese": japanese_chars / total_chars,
                "arabic": arabic_chars / total_chars
            }
            
            dominant_lang = max(ratios, key=ratios.get)
            confidence = ratios[dominant_lang]
            
            return {
                "language": dominant_lang,
                "confidence": confidence,
                "character_analysis": ratios
            }
            
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return {"language": "unknown", "confidence": 0}
    
    async def _calculate_confidence(self, engine_results: Dict[str, Dict]) -> float:
        """Calculate overall confidence score."""
        try:
            confidences = []
            for engine, result in engine_results.items():
                if "error" not in result:
                    confidences.append(result.get("avg_confidence", 0))
            
            if not confidences:
                return 0.0
            
            # Use weighted average, giving more weight to higher individual scores
            weights = [c/100 for c in confidences]  # Normalize to 0-1
            if sum(weights) > 0:
                weighted_avg = sum(c * w for c, w in zip(confidences, weights)) / sum(weights)
            else:
                weighted_avg = sum(confidences) / len(confidences)
            
            return float(weighted_avg)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    async def _determine_reading_order(self, layout_elements: List[Dict]) -> List[int]:
        """Determine reading order of layout elements."""
        try:
            if not layout_elements:
                return []
            
            # Sort elements by position (top-to-bottom, left-to-right)
            elements_with_idx = [(i, elem) for i, elem in enumerate(layout_elements)]
            
            # Sort by y-coordinate first (top to bottom), then x-coordinate (left to right)
            sorted_elements = sorted(elements_with_idx, 
                                   key=lambda x: (x[1]["bbox"][1], x[1]["bbox"][0]))
            
            reading_order = [elem[0] for elem in sorted_elements]
            
            return reading_order
            
        except Exception as e:
            logger.error(f"Error determining reading order: {e}")
            return list(range(len(layout_elements)))
    
    async def _detect_table_regions(self, image: np.ndarray) -> List[Dict]:
        """Detect table regions in image."""
        try:
            table_regions = []
            
            # Method 1: Using layout model if available
            if 'table' in self.layout_models:
                layout = self.layout_models['table'].detect(image)
                for element in layout:
                    if element.type.lower() == 'table':
                        table_regions.append({
                            "bbox": [int(element.x_1), int(element.y_1), 
                                   int(element.x_2), int(element.y_2)],
                            "confidence": float(element.score),
                            "method": "layout_model"
                        })
            
            # Method 2: Heuristic detection using lines
            if not table_regions:
                table_regions = await self._detect_tables_heuristic(image)
            
            return table_regions
            
        except Exception as e:
            logger.error(f"Error detecting table regions: {e}")
            return []
    
    async def _detect_tables_heuristic(self, image: np.ndarray) -> List[Dict]:
        """Detect tables using heuristic line detection."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect lines
            edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combine lines
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            table_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Filter by size and aspect ratio
                if area > 5000 and w > 100 and h > 50:
                    table_regions.append({
                        "bbox": [x, y, x + w, y + h],
                        "confidence": 0.7,  # Heuristic confidence
                        "method": "heuristic_lines"
                    })
            
            return table_regions
            
        except Exception as e:
            logger.error(f"Error in heuristic table detection: {e}")
            return []
    
    async def _extract_table_structure(self, table_image: np.ndarray) -> Dict[str, Any]:
        """Extract table structure and content."""
        try:
            # Detect lines to understand table structure
            if len(table_image.shape) == 3:
                gray = cv2.cvtColor(table_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = table_image
            
            # Detect horizontal and vertical lines
            edges = cv2.Canny(gray, 50, 150)
            
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # Find line coordinates
            h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=50, 
                                     minLineLength=100, maxLineGap=10)
            v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=50, 
                                     minLineLength=50, maxLineGap=5)
            
            # Estimate table dimensions
            rows = len(h_lines) - 1 if h_lines is not None else 1
            cols = len(v_lines) - 1 if v_lines is not None else 1
            
            # Extract cell contents using OCR
            cells = []
            if h_lines is not None and v_lines is not None:
                # Create grid of cell regions
                h_coords = sorted([line[0][1] for line in h_lines] + [line[0][3] for line in h_lines])
                v_coords = sorted([line[0][0] for line in v_lines] + [line[0][2] for line in v_lines])
                
                for i in range(len(h_coords) - 1):
                    for j in range(len(v_coords) - 1):
                        y1, y2 = h_coords[i], h_coords[i + 1]
                        x1, x2 = v_coords[j], v_coords[j + 1]
                        
                        # Extract cell image
                        cell_img = table_image[y1:y2, x1:x2]
                        
                        # OCR on cell
                        if 'easyocr' in self.engines:
                            cell_results = self.engines['easyocr'].readtext(cell_img)
                            cell_text = " ".join([result[1] for result in cell_results])
                        else:
                            cell_text = ""
                        
                        cells.append({
                            "row": i,
                            "col": j,
                            "bbox": [x1, y1, x2, y2],
                            "text": cell_text.strip()
                        })
            
            # Create data matrix
            data = []
            if cells:
                max_row = max(cell["row"] for cell in cells)
                max_col = max(cell["col"] for cell in cells)
                
                for r in range(max_row + 1):
                    row_data = []
                    for c in range(max_col + 1):
                        cell_text = ""
                        for cell in cells:
                            if cell["row"] == r and cell["col"] == c:
                                cell_text = cell["text"]
                                break
                        row_data.append(cell_text)
                    data.append(row_data)
            
            return {
                "rows": rows,
                "cols": cols,
                "cells": cells,
                "data": data
            }
            
        except Exception as e:
            logger.error(f"Error extracting table structure: {e}")
            return {"rows": 0, "cols": 0, "cells": [], "data": []}
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [
            "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"
        ]
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about available OCR engines."""
        return {
            "engines_available": list(self.engines.keys()),
            "layout_models_available": list(self.layout_models.keys()),
            "supported_languages": self.get_supported_languages(),
            "ocr_deps_available": HAS_OCR_DEPS
        }