"""
Advanced Image Processing Module

Provides comprehensive image processing capabilities including:
- Image enhancement and filtering
- Object detection and recognition
- Feature extraction and matching
- Image generation and manipulation
- Format conversion and optimization
"""

import asyncio
import base64
import io
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np

try:
    import cv2
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
    import torch
    import torchvision.transforms as transforms
    from ultralytics import YOLO
    from transformers import pipeline, AutoImageProcessor, AutoModel
    import clip
    from skimage import filters, segmentation, morphology, measure
    from scipy import ndimage
    HAS_VISION_DEPS = True
except ImportError:
    HAS_VISION_DEPS = False
    cv2 = None
    Image = None
    torch = None
    transforms = None
    YOLO = None

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Advanced image processing with AI-powered capabilities."""
    
    def __init__(self, model_cache_dir: Optional[str] = None):
        """Initialize image processor with optional model caching."""
        self.model_cache_dir = model_cache_dir or "./models/vision"
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not HAS_VISION_DEPS:
            logger.warning("Vision dependencies not installed. Limited functionality available.")
            return
            
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models for image processing."""
        try:
            # Object detection
            self.models['yolo'] = YOLO('yolov8n.pt')
            
            # CLIP for image-text understanding
            self.models['clip_model'], self.models['clip_preprocess'] = clip.load("ViT-B/32", device=self.device)
            
            # Image classification
            self.models['classifier'] = pipeline("image-classification", model="microsoft/resnet-50")
            
            # Image segmentation
            self.models['segmentation'] = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")
            
            logger.info("Image processing models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self.models = {}
    
    async def process_image(self, image_data: Union[str, bytes, np.ndarray], 
                          operations: List[str]) -> Dict[str, Any]:
        """
        Process image with specified operations.
        
        Args:
            image_data: Base64 string, bytes, or numpy array
            operations: List of operations to perform
        
        Returns:
            Dictionary with processing results
        """
        if not HAS_VISION_DEPS:
            return {"error": "Vision dependencies not installed"}
        
        try:
            # Load image
            image = await self._load_image(image_data)
            if image is None:
                return {"error": "Failed to load image"}
            
            results = {
                "original_size": image.shape[:2],
                "operations": operations,
                "results": {}
            }
            
            # Execute operations
            for operation in operations:
                result = await self._execute_operation(image, operation)
                results["results"][operation] = result
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"error": str(e)}
    
    async def _load_image(self, image_data: Union[str, bytes, np.ndarray]) -> Optional[np.ndarray]:
        """Load image from various input formats."""
        try:
            if isinstance(image_data, str):
                # Base64 string
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                return np.array(image)
                
            elif isinstance(image_data, bytes):
                # Raw bytes
                image = Image.open(io.BytesIO(image_data))
                return np.array(image)
                
            elif isinstance(image_data, np.ndarray):
                # Already a numpy array
                return image_data
                
            else:
                logger.error(f"Unsupported image data type: {type(image_data)}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    async def _execute_operation(self, image: np.ndarray, operation: str) -> Dict[str, Any]:
        """Execute a specific image processing operation."""
        try:
            if operation == "enhance":
                return await self._enhance_image(image)
            elif operation == "detect_objects":
                return await self._detect_objects(image)
            elif operation == "extract_features":
                return await self._extract_features(image)
            elif operation == "segment":
                return await self._segment_image(image)
            elif operation == "classify":
                return await self._classify_image(image)
            elif operation == "generate_caption":
                return await self._generate_caption(image)
            elif operation == "filter":
                return await self._apply_filters(image)
            elif operation == "analyze_colors":
                return await self._analyze_colors(image)
            elif operation == "extract_text_regions":
                return await self._extract_text_regions(image)
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            logger.error(f"Error executing operation {operation}: {e}")
            return {"error": str(e)}
    
    async def _enhance_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Enhance image quality using various techniques."""
        results = {}
        
        try:
            # Convert to PIL for enhancement
            pil_image = Image.fromarray(image)
            
            # Brightness enhancement
            enhancer = ImageEnhance.Brightness(pil_image)
            enhanced_brightness = enhancer.enhance(1.2)
            
            # Contrast enhancement  
            enhancer = ImageEnhance.Contrast(enhanced_brightness)
            enhanced_contrast = enhancer.enhance(1.1)
            
            # Sharpness enhancement
            enhancer = ImageEnhance.Sharpness(enhanced_contrast)
            enhanced_sharp = enhancer.enhance(1.1)
            
            # Color enhancement
            enhancer = ImageEnhance.Color(enhanced_sharp)
            enhanced_final = enhancer.enhance(1.05)
            
            # Convert back to numpy
            enhanced_array = np.array(enhanced_final)
            
            # Noise reduction using OpenCV
            denoised = cv2.bilateralFilter(enhanced_array, 9, 75, 75)
            
            # Histogram equalization for better contrast
            if len(denoised.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
                lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
                equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                equalized = cv2.equalizeHist(denoised)
            
            results = {
                "enhanced": True,
                "operations_applied": ["brightness", "contrast", "sharpness", "color", "denoise", "histogram_eq"],
                "quality_score": await self._calculate_quality_score(equalized),
                "enhanced_image": self._encode_image(equalized)
            }
            
        except Exception as e:
            results = {"error": f"Enhancement failed: {e}"}
        
        return results
    
    async def _detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect objects in image using YOLO."""
        if 'yolo' not in self.models:
            return {"error": "YOLO model not available"}
        
        try:
            # Run detection
            results = self.models['yolo'](image)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        detection = {
                            "class": result.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": box.xyxy[0].tolist(),
                            "center": box.xywh[0][:2].tolist()
                        }
                        detections.append(detection)
            
            return {
                "objects_detected": len(detections),
                "detections": detections,
                "classes": list(set([d["class"] for d in detections]))
            }
            
        except Exception as e:
            return {"error": f"Object detection failed: {e}"}
    
    async def _extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract visual features from image."""
        try:
            features = {}
            
            # Basic image statistics
            features["mean_intensity"] = float(np.mean(image))
            features["std_intensity"] = float(np.std(image))
            features["min_intensity"] = int(np.min(image))
            features["max_intensity"] = int(np.max(image))
            
            # Edge detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            edges = cv2.Canny(gray, 50, 150)
            features["edge_density"] = float(np.sum(edges > 0) / edges.size)
            
            # Texture features using GLCM
            from skimage.feature import graycomatrix, graycoprops
            glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
            features["contrast"] = float(graycoprops(glcm, 'contrast')[0, 0])
            features["dissimilarity"] = float(graycoprops(glcm, 'dissimilarity')[0, 0])
            features["homogeneity"] = float(graycoprops(glcm, 'homogeneity')[0, 0])
            
            # Shape features
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            features["num_contours"] = len(contours)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                features["largest_contour_area"] = float(cv2.contourArea(largest_contour))
                features["largest_contour_perimeter"] = float(cv2.arcLength(largest_contour, True))
            
            return features
            
        except Exception as e:
            return {"error": f"Feature extraction failed: {e}"}
    
    async def _segment_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Segment image into regions."""
        try:
            # Convert to PIL for transformer pipeline
            pil_image = Image.fromarray(image)
            
            if 'segmentation' in self.models:
                segments = self.models['segmentation'](pil_image)
                return {
                    "segments": len(segments),
                    "segment_info": segments
                }
            else:
                # Fallback to traditional segmentation
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                
                # Watershed segmentation
                from skimage.segmentation import watershed
                from skimage.feature import peak_local_maxima
                
                # Find local maxima
                local_maxima = peak_local_maxima(gray, min_distance=20)
                markers = np.zeros_like(gray)
                markers[tuple(local_maxima.T)] = np.arange(len(local_maxima)) + 1
                
                # Apply watershed
                segments = watershed(-gray, markers)
                
                return {
                    "segments": len(np.unique(segments)) - 1,
                    "method": "watershed",
                    "markers": len(local_maxima)
                }
                
        except Exception as e:
            return {"error": f"Segmentation failed: {e}"}
    
    async def _classify_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify image content."""
        try:
            if 'classifier' not in self.models:
                return {"error": "Classifier model not available"}
            
            # Convert to PIL
            pil_image = Image.fromarray(image)
            
            # Get predictions
            predictions = self.models['classifier'](pil_image)
            
            return {
                "predictions": predictions,
                "top_class": predictions[0]['label'],
                "confidence": predictions[0]['score']
            }
            
        except Exception as e:
            return {"error": f"Classification failed: {e}"}
    
    async def _generate_caption(self, image: np.ndarray) -> Dict[str, Any]:
        """Generate caption for image using CLIP."""
        try:
            if 'clip_model' not in self.models:
                return {"error": "CLIP model not available"}
            
            # Convert to PIL and preprocess
            pil_image = Image.fromarray(image)
            image_input = self.models['clip_preprocess'](pil_image).unsqueeze(0).to(self.device)
            
            # Predefined captions to choose from
            captions = [
                "a photo of a person",
                "a photo of an animal", 
                "a photo of a building",
                "a photo of food",
                "a photo of nature",
                "a photo of a vehicle",
                "a photo of technology",
                "a photo of art",
                "a photo of sports",
                "a indoor scene",
                "an outdoor scene"
            ]
            
            text_inputs = clip.tokenize(captions).to(self.device)
            
            with torch.no_grad():
                image_features = self.models['clip_model'].encode_image(image_input)
                text_features = self.models['clip_model'].encode_text(text_inputs)
                
                # Calculate similarities
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarities[0].topk(3)
            
            results = []
            for i, (value, index) in enumerate(zip(values, indices)):
                results.append({
                    "caption": captions[index],
                    "confidence": float(value)
                })
            
            return {
                "captions": results,
                "best_caption": results[0]["caption"]
            }
            
        except Exception as e:
            return {"error": f"Caption generation failed: {e}"}
    
    async def _apply_filters(self, image: np.ndarray) -> Dict[str, Any]:
        """Apply various image filters."""
        try:
            filters_applied = {}
            
            # Gaussian blur
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            filters_applied["gaussian_blur"] = self._encode_image(blurred)
            
            # Edge detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            edges = cv2.Canny(gray, 50, 150)
            if len(image.shape) == 3:
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            filters_applied["edges"] = self._encode_image(edges)
            
            # Emboss filter
            kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
            embossed = cv2.filter2D(image, -1, kernel)
            filters_applied["emboss"] = self._encode_image(embossed)
            
            # Sharpen filter
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            sharpened = cv2.filter2D(image, -1, kernel)
            filters_applied["sharpen"] = self._encode_image(sharpened)
            
            return {
                "filters_applied": list(filters_applied.keys()),
                "filtered_images": filters_applied
            }
            
        except Exception as e:
            return {"error": f"Filter application failed: {e}"}
    
    async def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution in image."""
        try:
            # Convert to appropriate color space
            if len(image.shape) == 3:
                # RGB analysis
                colors = image.reshape(-1, 3)
                
                # Dominant colors using k-means clustering
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=5, random_state=42)
                kmeans.fit(colors)
                
                dominant_colors = kmeans.cluster_centers_.astype(int)
                color_percentages = np.bincount(kmeans.labels_) / len(kmeans.labels_)
                
                color_analysis = {
                    "dominant_colors": [
                        {
                            "rgb": color.tolist(),
                            "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                            "percentage": float(percentage)
                        }
                        for color, percentage in zip(dominant_colors, color_percentages)
                    ],
                    "color_diversity": float(np.std(colors)),
                    "brightness": float(np.mean(colors)),
                    "contrast": float(np.std(colors))
                }
                
                # Color temperature analysis
                avg_color = np.mean(colors, axis=0)
                if avg_color[0] > avg_color[2]:
                    temperature = "warm"
                else:
                    temperature = "cool"
                    
                color_analysis["temperature"] = temperature
                
            else:
                # Grayscale analysis
                color_analysis = {
                    "type": "grayscale",
                    "histogram": np.histogram(image, bins=256)[0].tolist(),
                    "mean_intensity": float(np.mean(image)),
                    "contrast": float(np.std(image))
                }
            
            return color_analysis
            
        except Exception as e:
            return {"error": f"Color analysis failed: {e}"}
    
    async def _extract_text_regions(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text regions from image."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Text detection using MSER (Maximally Stable Extremal Regions)
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            text_regions = []
            for region in regions:
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                # Filter by aspect ratio and size to likely text regions
                aspect_ratio = w / h
                if 0.1 < aspect_ratio < 10 and w > 10 and h > 10:
                    text_regions.append({
                        "bbox": [x, y, x + w, y + h],
                        "width": w,
                        "height": h,
                        "aspect_ratio": aspect_ratio
                    })
            
            # Additional text detection using edge-based method
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            additional_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 8 and w > 20 and h > 8:
                    additional_regions.append({
                        "bbox": [x, y, x + w, y + h],
                        "width": w,
                        "height": h,
                        "aspect_ratio": aspect_ratio,
                        "method": "edge_based"
                    })
            
            return {
                "mser_regions": len(text_regions),
                "edge_regions": len(additional_regions),
                "total_regions": len(text_regions) + len(additional_regions),
                "text_regions": text_regions + additional_regions
            }
            
        except Exception as e:
            return {"error": f"Text region extraction failed: {e}"}
    
    async def _calculate_quality_score(self, image: np.ndarray) -> float:
        """Calculate image quality score."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Laplacian variance (focus measure)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-100 scale
            quality_score = min(100, laplacian_var / 1000 * 100)
            
            return float(quality_score)
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.0
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image as base64 string."""
        try:
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return ""
    
    async def batch_process(self, images: List[Union[str, bytes, np.ndarray]], 
                          operations: List[str]) -> List[Dict[str, Any]]:
        """Process multiple images in batch."""
        tasks = [self.process_image(image, operations) for image in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            result if not isinstance(result, Exception) else {"error": str(result)}
            for result in results
        ]
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported image processing operations."""
        return [
            "enhance",
            "detect_objects", 
            "extract_features",
            "segment",
            "classify",
            "generate_caption",
            "filter",
            "analyze_colors",
            "extract_text_regions"
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "models_loaded": list(self.models.keys()),
            "device": self.device,
            "vision_deps_available": HAS_VISION_DEPS,
            "supported_operations": self.get_supported_operations()
        }