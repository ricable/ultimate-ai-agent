"""
Document Layout Analysis Module

Provides advanced document layout analysis capabilities including:
- Document structure detection and analysis
- Table detection and extraction
- Reading order determination
- Text region classification
- Form field detection
- Document type classification
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
import json

try:
    import cv2
    from PIL import Image, ImageDraw
    import layoutparser as lp
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    from scipy import ndimage
    from sklearn.cluster import DBSCAN
    HAS_LAYOUT_DEPS = True
except ImportError:
    HAS_LAYOUT_DEPS = False
    cv2 = None
    Image = None
    lp = None
    torch = None

logger = logging.getLogger(__name__)

class DocumentLayoutAnalyzer:
    """Advanced document layout analysis with AI-powered capabilities."""
    
    def __init__(self, model_cache_dir: Optional[str] = None):
        """Initialize document layout analyzer."""
        self.model_cache_dir = model_cache_dir or "./models/layout"
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not HAS_LAYOUT_DEPS:
            logger.warning("Layout analysis dependencies not installed. Limited functionality available.")
            return
            
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize layout analysis models."""
        try:
            # LayoutParser models for different document types
            self.models['newspaper'] = lp.AutoLayoutModel(
                'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
            )
            
            self.models['table'] = lp.AutoLayoutModel(
                'lp://TableBank/faster_rcnn_R_50_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
            )
            
            self.models['form'] = lp.AutoLayoutModel(
                'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
            )
            
            # Document classification pipeline
            self.models['doc_classifier'] = pipeline(
                "image-classification",
                model="microsoft/dit-base-finetuned-rvlcdip"
            )
            
            logger.info("Document layout models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing layout models: {e}")
            self.models = {}
    
    async def analyze_document_layout(self, image_data: Union[str, bytes, np.ndarray],
                                    document_type: str = "auto") -> Dict[str, Any]:
        """
        Analyze document layout and structure.
        
        Args:
            image_data: Document image data
            document_type: Type of document ("auto", "newspaper", "form", "table")
        
        Returns:
            Dictionary with layout analysis results
        """
        if not HAS_LAYOUT_DEPS:
            return {"error": "Layout analysis dependencies not installed"}
        
        try:
            # Load image
            image = await self._load_image(image_data)
            if image is None:
                return {"error": "Failed to load image"}
            
            results = {
                "image_size": image.shape[:2],
                "document_type": document_type,
                "layout_elements": [],
                "statistics": {},
                "reading_order": [],
                "metadata": {}
            }
            
            # Auto-detect document type if needed
            if document_type == "auto":
                document_type = await self._classify_document_type(image)
                results["document_type"] = document_type
            
            # Select appropriate model
            model_key = self._get_model_key(document_type)
            if model_key not in self.models:
                model_key = 'newspaper'  # Default fallback
            
            # Detect layout elements
            layout = self.models[model_key].detect(image)
            
            # Process detected elements
            for element in layout:
                element_info = {
                    "type": element.type,
                    "bbox": [int(element.x_1), int(element.y_1), int(element.x_2), int(element.y_2)],
                    "confidence": float(element.score),
                    "area": int((element.x_2 - element.x_1) * (element.y_2 - element.y_1)),
                    "center": [
                        int((element.x_1 + element.x_2) / 2),
                        int((element.y_1 + element.y_2) / 2)
                    ]
                }
                results["layout_elements"].append(element_info)
            
            # Calculate statistics
            results["statistics"] = await self._calculate_layout_statistics(results["layout_elements"])
            
            # Determine reading order
            results["reading_order"] = await self._determine_reading_order(results["layout_elements"])
            
            # Extract metadata
            results["metadata"] = await self._extract_document_metadata(image, results["layout_elements"])
            
            # Detect tables if present
            if any(elem["type"].lower() == "table" for elem in results["layout_elements"]):
                results["table_analysis"] = await self._analyze_tables(image, results["layout_elements"])
            
            # Detect forms if present
            if document_type in ["form", "auto"]:
                results["form_analysis"] = await self._analyze_forms(image, results["layout_elements"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing document layout: {e}")
            return {"error": str(e)}
    
    async def extract_text_regions(self, image_data: Union[str, bytes, np.ndarray],
                                 classify_regions: bool = True) -> Dict[str, Any]:
        """
        Extract and classify text regions from document.
        
        Args:
            image_data: Document image data
            classify_regions: Whether to classify text region types
        
        Returns:
            Dictionary with text region information
        """
        if not HAS_LAYOUT_DEPS:
            return {"error": "Text region extraction not available"}
        
        try:
            # Load image
            image = await self._load_image(image_data)
            if image is None:
                return {"error": "Failed to load image"}
            
            # Detect layout elements
            layout = self.models['newspaper'].detect(image)
            
            text_regions = []
            for element in layout:
                if element.type.lower() in ['text', 'title', 'list']:
                    region_info = {
                        "type": element.type,
                        "bbox": [int(element.x_1), int(element.y_1), int(element.x_2), int(element.y_2)],
                        "confidence": float(element.score),
                        "area": int((element.x_2 - element.x_1) * (element.y_2 - element.y_1))
                    }
                    
                    # Extract region image for further analysis
                    region_image = image[int(element.y_1):int(element.y_2), 
                                       int(element.x_1):int(element.x_2)]
                    
                    # Classify text region if requested
                    if classify_regions:
                        region_info["classification"] = await self._classify_text_region(region_image)
                    
                    text_regions.append(region_info)
            
            # Group nearby regions
            grouped_regions = await self._group_text_regions(text_regions)
            
            return {
                "text_regions": text_regions,
                "grouped_regions": grouped_regions,
                "total_regions": len(text_regions),
                "region_types": list(set(region["type"] for region in text_regions))
            }
            
        except Exception as e:
            logger.error(f"Error extracting text regions: {e}")
            return {"error": str(e)}
    
    async def detect_tables(self, image_data: Union[str, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Detect and analyze tables in document.
        
        Args:
            image_data: Document image data
        
        Returns:
            Dictionary with table detection results
        """
        if not HAS_LAYOUT_DEPS or 'table' not in self.models:
            return {"error": "Table detection not available"}
        
        try:
            # Load image
            image = await self._load_image(image_data)
            if image is None:
                return {"error": "Failed to load image"}
            
            # Detect tables
            layout = self.models['table'].detect(image)
            
            tables = []
            for i, element in enumerate(layout):
                if element.type.lower() == 'table':
                    # Extract table region
                    table_region = image[int(element.y_1):int(element.y_2), 
                                       int(element.x_1):int(element.x_2)]
                    
                    # Analyze table structure
                    table_analysis = await self._analyze_table_structure(table_region)
                    
                    table_info = {
                        "table_id": i,
                        "bbox": [int(element.x_1), int(element.y_1), int(element.x_2), int(element.y_2)],
                        "confidence": float(element.score),
                        "structure": table_analysis
                    }
                    
                    tables.append(table_info)
            
            return {
                "tables_detected": len(tables),
                "tables": tables
            }
            
        except Exception as e:
            logger.error(f"Error detecting tables: {e}")
            return {"error": str(e)}
    
    async def analyze_form_fields(self, image_data: Union[str, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Detect and analyze form fields in document.
        
        Args:
            image_data: Document image data
        
        Returns:
            Dictionary with form field analysis
        """
        if not HAS_LAYOUT_DEPS:
            return {"error": "Form analysis not available"}
        
        try:
            # Load image
            image = await self._load_image(image_data)
            if image is None:
                return {"error": "Failed to load image"}
            
            # Convert to grayscale for form field detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            form_fields = []
            
            # Detect rectangular regions (potential form fields)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Filter by size and aspect ratio for form fields
                if area > 500 and w > 50 and h > 10:
                    aspect_ratio = w / h
                    
                    # Classify field type based on dimensions
                    if aspect_ratio > 5:
                        field_type = "text_field"
                    elif aspect_ratio < 2 and w < 100 and h < 100:
                        field_type = "checkbox"
                    elif aspect_ratio > 2 and aspect_ratio < 5:
                        field_type = "input_field"
                    else:
                        field_type = "unknown"
                    
                    form_fields.append({
                        "field_id": i,
                        "type": field_type,
                        "bbox": [x, y, x + w, y + h],
                        "area": int(area),
                        "aspect_ratio": aspect_ratio
                    })
            
            # Group fields into logical sections
            sections = await self._group_form_fields(form_fields)
            
            return {
                "form_fields": form_fields,
                "sections": sections,
                "total_fields": len(form_fields),
                "field_types": list(set(field["type"] for field in form_fields))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing form fields: {e}")
            return {"error": str(e)}
    
    async def _load_image(self, image_data: Union[str, bytes, np.ndarray]) -> Optional[np.ndarray]:
        """Load image from various input formats."""
        try:
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    import base64
                    import io
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
    
    async def _classify_document_type(self, image: np.ndarray) -> str:
        """Classify document type."""
        try:
            if 'doc_classifier' in self.models:
                pil_image = Image.fromarray(image)
                classification = self.models['doc_classifier'](pil_image)
                return classification[0]['label'].lower()
            else:
                # Fallback heuristic classification
                return await self._heuristic_document_classification(image)
                
        except Exception as e:
            logger.error(f"Error classifying document: {e}")
            return "document"
    
    async def _heuristic_document_classification(self, image: np.ndarray) -> str:
        """Heuristic document type classification."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect lines for table/form detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # Count lines
            h_line_count = np.sum(horizontal_lines > 0)
            v_line_count = np.sum(vertical_lines > 0)
            
            # Classification based on line density
            total_pixels = gray.shape[0] * gray.shape[1]
            line_density = (h_line_count + v_line_count) / total_pixels
            
            if line_density > 0.01:
                return "form" if h_line_count < v_line_count else "table"
            else:
                return "article"
                
        except Exception as e:
            logger.error(f"Error in heuristic classification: {e}")
            return "document"
    
    def _get_model_key(self, document_type: str) -> str:
        """Get appropriate model key for document type."""
        mapping = {
            "newspaper": "newspaper",
            "article": "newspaper", 
            "form": "form",
            "table": "table",
            "document": "newspaper"
        }
        return mapping.get(document_type, "newspaper")
    
    async def _calculate_layout_statistics(self, layout_elements: List[Dict]) -> Dict[str, Any]:
        """Calculate layout statistics."""
        try:
            if not layout_elements:
                return {"total_elements": 0}
            
            # Element type counts
            element_types = [elem["type"] for elem in layout_elements]
            type_counts = {elem_type: element_types.count(elem_type) 
                          for elem_type in set(element_types)}
            
            # Area statistics
            areas = [elem["area"] for elem in layout_elements]
            
            # Spatial distribution
            centers_x = [elem["center"][0] for elem in layout_elements]
            centers_y = [elem["center"][1] for elem in layout_elements]
            
            return {
                "total_elements": len(layout_elements),
                "element_types": type_counts,
                "area_statistics": {
                    "mean_area": np.mean(areas),
                    "total_area": sum(areas),
                    "largest_element": max(areas) if areas else 0,
                    "smallest_element": min(areas) if areas else 0
                },
                "spatial_distribution": {
                    "horizontal_spread": max(centers_x) - min(centers_x) if centers_x else 0,
                    "vertical_spread": max(centers_y) - min(centers_y) if centers_y else 0,
                    "center_of_mass": [np.mean(centers_x), np.mean(centers_y)] if centers_x else [0, 0]
                },
                "confidence_statistics": {
                    "mean_confidence": np.mean([elem["confidence"] for elem in layout_elements]),
                    "min_confidence": min([elem["confidence"] for elem in layout_elements]),
                    "max_confidence": max([elem["confidence"] for elem in layout_elements])
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {"error": str(e)}
    
    async def _determine_reading_order(self, layout_elements: List[Dict]) -> List[int]:
        """Determine reading order of layout elements."""
        try:
            if not layout_elements:
                return []
            
            # Sort elements by position (top-to-bottom, left-to-right)
            elements_with_idx = [(i, elem) for i, elem in enumerate(layout_elements)]
            
            # Primary sort by y-coordinate (top to bottom)
            # Secondary sort by x-coordinate (left to right)
            sorted_elements = sorted(elements_with_idx, 
                                   key=lambda x: (x[1]["bbox"][1], x[1]["bbox"][0]))
            
            reading_order = [elem[0] for elem in sorted_elements]
            
            return reading_order
            
        except Exception as e:
            logger.error(f"Error determining reading order: {e}")
            return list(range(len(layout_elements)))
    
    async def _extract_document_metadata(self, image: np.ndarray, 
                                       layout_elements: List[Dict]) -> Dict[str, Any]:
        """Extract document metadata."""
        try:
            metadata = {
                "page_dimensions": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                },
                "layout_complexity": len(layout_elements),
                "dominant_elements": [],
                "text_density": 0.0
            }
            
            # Find dominant element types
            if layout_elements:
                element_types = [elem["type"] for elem in layout_elements]
                type_counts = {elem_type: element_types.count(elem_type) 
                              for elem_type in set(element_types)}
                
                # Sort by count
                sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
                metadata["dominant_elements"] = [elem_type for elem_type, count in sorted_types[:3]]
                
                # Calculate text density
                text_elements = [elem for elem in layout_elements if elem["type"].lower() in ['text', 'title']]
                text_area = sum(elem["area"] for elem in text_elements)
                total_area = image.shape[0] * image.shape[1]
                metadata["text_density"] = text_area / total_area if total_area > 0 else 0
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {"error": str(e)}
    
    async def _analyze_tables(self, image: np.ndarray, layout_elements: List[Dict]) -> Dict[str, Any]:
        """Analyze table structures in document."""
        try:
            table_elements = [elem for elem in layout_elements if elem["type"].lower() == "table"]
            
            if not table_elements:
                return {"tables_found": 0}
            
            table_analysis = []
            
            for i, table_elem in enumerate(table_elements):
                # Extract table region
                bbox = table_elem["bbox"]
                table_region = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                
                # Analyze table structure
                structure = await self._analyze_table_structure(table_region)
                
                table_analysis.append({
                    "table_id": i,
                    "bbox": bbox,
                    "structure": structure
                })
            
            return {
                "tables_found": len(table_elements),
                "table_analysis": table_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing tables: {e}")
            return {"error": str(e)}
    
    async def _analyze_table_structure(self, table_image: np.ndarray) -> Dict[str, Any]:
        """Analyze individual table structure."""
        try:
            if len(table_image.shape) == 3:
                gray = cv2.cvtColor(table_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = table_image
            
            # Detect lines
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # Find line coordinates
            h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=50)
            v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=50)
            
            # Estimate table dimensions
            estimated_rows = len(h_lines) - 1 if h_lines is not None else 1
            estimated_cols = len(v_lines) - 1 if v_lines is not None else 1
            
            return {
                "estimated_rows": max(1, estimated_rows),
                "estimated_cols": max(1, estimated_cols),
                "has_borders": h_lines is not None and v_lines is not None,
                "border_quality": "good" if (h_lines is not None and v_lines is not None) else "poor"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing table structure: {e}")
            return {"estimated_rows": 1, "estimated_cols": 1, "has_borders": False}
    
    async def _analyze_forms(self, image: np.ndarray, layout_elements: List[Dict]) -> Dict[str, Any]:
        """Analyze form elements in document."""
        try:
            # This is a simplified form analysis
            # In practice, you'd want more sophisticated form field detection
            
            form_elements = []
            
            # Look for potential form fields based on layout elements
            for elem in layout_elements:
                if elem["type"].lower() in ["text", "figure"]:
                    # Check if element could be a form field
                    bbox = elem["bbox"]
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    aspect_ratio = width / height if height > 0 else 0
                    
                    # Heuristic for form field detection
                    if 2 < aspect_ratio < 20 and height < 100:
                        form_elements.append({
                            "type": "potential_field",
                            "bbox": bbox,
                            "aspect_ratio": aspect_ratio
                        })
            
            return {
                "potential_form_fields": len(form_elements),
                "form_elements": form_elements
            }
            
        except Exception as e:
            logger.error(f"Error analyzing forms: {e}")
            return {"error": str(e)}
    
    async def _classify_text_region(self, region_image: np.ndarray) -> Dict[str, Any]:
        """Classify text region type."""
        try:
            # Simple classification based on region properties
            height, width = region_image.shape[:2]
            aspect_ratio = width / height if height > 0 else 0
            
            # Heuristic classification
            if aspect_ratio > 10:
                region_type = "header"
            elif aspect_ratio < 2:
                region_type = "column_text"
            elif height > 200:
                region_type = "paragraph"
            else:
                region_type = "text_block"
            
            return {
                "classified_type": region_type,
                "confidence": 0.7,  # Heuristic confidence
                "properties": {
                    "aspect_ratio": aspect_ratio,
                    "width": width,
                    "height": height
                }
            }
            
        except Exception as e:
            logger.error(f"Error classifying text region: {e}")
            return {"classified_type": "unknown", "confidence": 0}
    
    async def _group_text_regions(self, text_regions: List[Dict]) -> List[List[int]]:
        """Group nearby text regions."""
        try:
            if len(text_regions) < 2:
                return [[i] for i in range(len(text_regions))]
            
            # Extract centers for clustering
            centers = np.array([[
                (region["bbox"][0] + region["bbox"][2]) / 2,
                (region["bbox"][1] + region["bbox"][3]) / 2
            ] for region in text_regions])
            
            # Use DBSCAN for clustering
            clustering = DBSCAN(eps=100, min_samples=1).fit(centers)
            labels = clustering.labels_
            
            # Group by cluster labels
            groups = {}
            for i, label in enumerate(labels):
                if label not in groups:
                    groups[label] = []
                groups[label].append(i)
            
            return list(groups.values())
            
        except Exception as e:
            logger.error(f"Error grouping text regions: {e}")
            return [[i] for i in range(len(text_regions))]
    
    async def _group_form_fields(self, form_fields: List[Dict]) -> List[Dict]:
        """Group form fields into logical sections."""
        try:
            if not form_fields:
                return []
            
            # Simple vertical grouping based on y-coordinates
            sorted_fields = sorted(form_fields, key=lambda x: x["bbox"][1])
            
            sections = []
            current_section = []
            last_y = None
            section_threshold = 50  # Pixels
            
            for field in sorted_fields:
                field_y = field["bbox"][1]
                
                if last_y is None or abs(field_y - last_y) < section_threshold:
                    current_section.append(field["field_id"])
                else:
                    if current_section:
                        sections.append({
                            "section_id": len(sections),
                            "fields": current_section.copy(),
                            "field_count": len(current_section)
                        })
                    current_section = [field["field_id"]]
                
                last_y = field_y
            
            # Add last section
            if current_section:
                sections.append({
                    "section_id": len(sections),
                    "fields": current_section,
                    "field_count": len(current_section)
                })
            
            return sections
            
        except Exception as e:
            logger.error(f"Error grouping form fields: {e}")
            return []
    
    def get_supported_document_types(self) -> List[str]:
        """Get list of supported document types."""
        return ["auto", "newspaper", "article", "form", "table", "document"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "models_loaded": list(self.models.keys()),
            "device": self.device,
            "layout_deps_available": HAS_LAYOUT_DEPS,
            "supported_document_types": self.get_supported_document_types()
        }