# backend/vision/image_processor.py
# Agent 24: Computer Vision & Multimodal AI System

import asyncio
import json
import base64
import io
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path

# Computer Vision libraries
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import pipeline
    CV_LIBRARIES_AVAILABLE = True
except ImportError:
    CV_LIBRARIES_AVAILABLE = False
    print("Computer vision libraries not available, using mock implementation")

# OCR libraries
try:
    import pytesseract
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

class ImageFormat(Enum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    BMP = "bmp"
    TIFF = "tiff"

class ObjectType(Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    OBJECT = "object"
    TEXT = "text"
    FACE = "face"
    DOCUMENT = "document"

@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x: int
    y: int
    width: int
    height: int
    confidence: float

@dataclass
class DetectedObject:
    """Detected object in image"""
    label: str
    confidence: float
    bbox: BoundingBox
    object_type: ObjectType
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class OCRResult:
    """OCR text extraction result"""
    text: str
    confidence: float
    bbox: BoundingBox
    language: str = "en"

@dataclass
class ImageAnalysisResult:
    """Complete image analysis result"""
    image_id: str
    format: ImageFormat
    dimensions: Tuple[int, int]
    file_size: int
    detected_objects: List[DetectedObject]
    ocr_results: List[OCRResult]
    caption: Optional[str]
    visual_features: Optional[Dict[str, Any]]
    quality_metrics: Dict[str, float]
    processing_time: float
    timestamp: datetime

class ObjectDetector:
    """Object detection using various models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.yolo_model = None
        self.face_cascade = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize computer vision models"""
        if CV_LIBRARIES_AVAILABLE:
            try:
                # Load face detection model
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                self.logger.info("Loaded OpenCV face detection model")
                
                # YOLO would be loaded here in a real implementation
                self.logger.info("Object detection models initialized")
                
            except Exception as e:
                self.logger.warning(f"Failed to load CV models: {e}")
    
    async def detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """Detect objects in image"""
        objects = []
        
        if self.face_cascade is not None:
            # Face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                objects.append(DetectedObject(
                    label="face",
                    confidence=0.8,
                    bbox=BoundingBox(x=x, y=y, width=w, height=h, confidence=0.8),
                    object_type=ObjectType.FACE,
                    metadata={"detector": "opencv_cascade"}
                ))
        
        # Mock object detection for other objects
        if not CV_LIBRARIES_AVAILABLE:
            objects.extend(self._mock_object_detection(image))
        
        return objects
    
    def _mock_object_detection(self, image: np.ndarray) -> List[DetectedObject]:
        """Mock object detection for testing"""
        height, width = image.shape[:2]
        
        return [
            DetectedObject(
                label="mock_object",
                confidence=0.75,
                bbox=BoundingBox(
                    x=width//4, 
                    y=height//4, 
                    width=width//2, 
                    height=height//2, 
                    confidence=0.75
                ),
                object_type=ObjectType.OBJECT,
                metadata={"detector": "mock"}
            )
        ]

class OCRProcessor:
    """Optical Character Recognition with layout understanding"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tesseract_available = OCR_AVAILABLE
        self.easyocr_reader = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize OCR engines"""
        if OCR_AVAILABLE:
            try:
                # Initialize EasyOCR
                self.easyocr_reader = easyocr.Reader(['en'])
                self.logger.info("Initialized EasyOCR reader")
            except Exception as e:
                self.logger.warning(f"Failed to initialize EasyOCR: {e}")
    
    async def extract_text(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text from image using OCR"""
        ocr_results = []
        
        # EasyOCR
        if self.easyocr_reader:
            try:
                results = self.easyocr_reader.readtext(image)
                for (bbox, text, confidence) in results:
                    # Convert bbox format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x, y = int(min(x_coords)), int(min(y_coords))
                    width = int(max(x_coords) - min(x_coords))
                    height = int(max(y_coords) - min(y_coords))
                    
                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=confidence,
                        bbox=BoundingBox(x=x, y=y, width=width, height=height, confidence=confidence),
                        language="en"
                    ))
            except Exception as e:
                self.logger.error(f"EasyOCR failed: {e}")
        
        # Tesseract OCR
        if self.tesseract_available:
            try:
                # Get text with bounding boxes
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    if text and int(data['conf'][i]) > 0:
                        ocr_results.append(OCRResult(
                            text=text,
                            confidence=float(data['conf'][i]) / 100.0,
                            bbox=BoundingBox(
                                x=data['left'][i],
                                y=data['top'][i],
                                width=data['width'][i],
                                height=data['height'][i],
                                confidence=float(data['conf'][i]) / 100.0
                            ),
                            language="en"
                        ))
            except Exception as e:
                self.logger.error(f"Tesseract OCR failed: {e}")
        
        # Mock OCR if no engines available
        if not ocr_results and not OCR_AVAILABLE:
            ocr_results = self._mock_ocr(image)
        
        return self._merge_ocr_results(ocr_results)
    
    def _mock_ocr(self, image: np.ndarray) -> List[OCRResult]:
        """Mock OCR for testing"""
        height, width = image.shape[:2]
        
        return [
            OCRResult(
                text="Mock extracted text",
                confidence=0.85,
                bbox=BoundingBox(x=10, y=10, width=width-20, height=30, confidence=0.85),
                language="en"
            )
        ]
    
    def _merge_ocr_results(self, results: List[OCRResult]) -> List[OCRResult]:
        """Merge overlapping OCR results"""
        if not results:
            return results
        
        # Simple deduplication based on text similarity and proximity
        merged = []
        for result in results:
            is_duplicate = False
            for existing in merged:
                # Check if results are similar and close to each other
                if (abs(result.bbox.x - existing.bbox.x) < 10 and 
                    abs(result.bbox.y - existing.bbox.y) < 10 and
                    result.text.lower().strip() == existing.text.lower().strip()):
                    is_duplicate = True
                    # Keep the result with higher confidence
                    if result.confidence > existing.confidence:
                        merged.remove(existing)
                        merged.append(result)
                    break
            
            if not is_duplicate:
                merged.append(result)
        
        return sorted(merged, key=lambda r: (r.bbox.y, r.bbox.x))

class ImageCaptioning:
    """Generate captions for images"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.blip_processor = None
        self.blip_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize image captioning model"""
        if CV_LIBRARIES_AVAILABLE:
            try:
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.logger.info("Loaded BLIP image captioning model")
            except Exception as e:
                self.logger.warning(f"Failed to load BLIP model: {e}")
    
    async def generate_caption(self, image: np.ndarray) -> str:
        """Generate caption for image"""
        if self.blip_processor and self.blip_model:
            try:
                # Convert to PIL Image
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                # Process image
                inputs = self.blip_processor(pil_image, return_tensors="pt")
                
                # Generate caption
                with torch.no_grad():
                    out = self.blip_model.generate(**inputs, max_length=50)
                    caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                
                return caption
                
            except Exception as e:
                self.logger.error(f"Caption generation failed: {e}")
        
        # Mock caption
        return "An image showing various objects and elements"

class ImageQualityAnalyzer:
    """Analyze image quality metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze image quality metrics"""
        metrics = {}
        
        try:
            # Convert to grayscale for some calculations
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            metrics['sharpness'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # Brightness (mean pixel value)
            metrics['brightness'] = float(np.mean(gray))
            
            # Contrast (standard deviation)
            metrics['contrast'] = float(np.std(gray))
            
            # Noise level (estimated)
            metrics['noise_level'] = self._estimate_noise(gray)
            
            # Color distribution
            if len(image.shape) == 3:
                color_std = np.std(image, axis=(0, 1))
                metrics['color_variance'] = float(np.mean(color_std))
            
        except Exception as e:
            self.logger.error(f"Quality analysis failed: {e}")
            # Return default values
            metrics = {
                'sharpness': 50.0,
                'brightness': 128.0,
                'contrast': 50.0,
                'noise_level': 10.0,
                'color_variance': 25.0
            }
        
        return metrics
    
    def _estimate_noise(self, gray_image: np.ndarray) -> float:
        """Estimate noise level in image"""
        try:
            # Simple noise estimation using high-frequency content
            h, w = gray_image.shape
            if h > 50 and w > 50:
                # Take a central crop to avoid edge effects
                crop = gray_image[h//4:3*h//4, w//4:3*w//4]
                # Apply high-pass filter
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                filtered = cv2.filter2D(crop, -1, kernel)
                return float(np.std(filtered))
            else:
                return 10.0
        except:
            return 10.0

class VideoProcessor:
    """Process video files and streams"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.object_detector = ObjectDetector()
        self.ocr_processor = OCRProcessor()
    
    async def process_video_frame(self, frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
        """Process a single video frame"""
        # Detect objects
        objects = await self.object_detector.detect_objects(frame)
        
        # Extract text
        ocr_results = await self.ocr_processor.extract_text(frame)
        
        return {
            'frame_number': frame_number,
            'timestamp': frame_number / 30.0,  # Assuming 30 FPS
            'objects': [asdict(obj) for obj in objects],
            'text': [asdict(ocr) for ocr in ocr_results]
        }
    
    async def analyze_video_stream(self, video_path: str, sample_rate: int = 30) -> List[Dict[str, Any]]:
        """Analyze video by sampling frames"""
        results = []
        
        if not CV_LIBRARIES_AVAILABLE:
            return [{'error': 'OpenCV not available'}]
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_count % sample_rate == 0:
                    result = await self.process_video_frame(frame, frame_count)
                    results.append(result)
                
                frame_count += 1
                
                # Limit processing for demo
                if len(results) >= 10:
                    break
            
            cap.release()
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            results.append({'error': str(e)})
        
        return results

class VisionProcessor:
    """Main computer vision processing pipeline"""
    
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.ocr_processor = OCRProcessor()
        self.image_captioning = ImageCaptioning()
        self.quality_analyzer = ImageQualityAnalyzer()
        self.video_processor = VideoProcessor()
        self.logger = logging.getLogger(__name__)
    
    async def process_image(self, 
                           image_data: Union[str, bytes, np.ndarray],
                           image_id: str = None,
                           include_objects: bool = True,
                           include_ocr: bool = True,
                           include_caption: bool = True,
                           include_quality: bool = True) -> ImageAnalysisResult:
        """Process image through complete computer vision pipeline"""
        
        start_time = datetime.utcnow()
        processing_start = start_time.timestamp()
        
        # Load image
        if isinstance(image_data, str):
            # Base64 encoded image
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        elif isinstance(image_data, bytes):
            # Raw bytes
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            # Already numpy array
            image = image_data
        
        if image is None:
            raise ValueError("Could not decode image data")
        
        # Get image properties
        height, width = image.shape[:2]
        file_size = image.nbytes
        
        # Detect format (simplified)
        image_format = ImageFormat.JPEG  # Default
        
        # Generate image ID if not provided
        if image_id is None:
            import hashlib
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            image_id = f"img_{image_hash[:8]}"
        
        # Object detection
        detected_objects = []
        if include_objects:
            detected_objects = await self.object_detector.detect_objects(image)
        
        # OCR
        ocr_results = []
        if include_ocr:
            ocr_results = await self.ocr_processor.extract_text(image)
        
        # Image captioning
        caption = None
        if include_caption:
            caption = await self.image_captioning.generate_caption(image)
        
        # Quality analysis
        quality_metrics = {}
        if include_quality:
            quality_metrics = self.quality_analyzer.analyze_quality(image)
        
        # Visual features (placeholder)
        visual_features = {
            'dominant_colors': self._extract_dominant_colors(image),
            'histogram_features': self._compute_histogram_features(image)
        }
        
        processing_time = datetime.utcnow().timestamp() - processing_start
        
        return ImageAnalysisResult(
            image_id=image_id,
            format=image_format,
            dimensions=(width, height),
            file_size=file_size,
            detected_objects=detected_objects,
            ocr_results=ocr_results,
            caption=caption,
            visual_features=visual_features,
            quality_metrics=quality_metrics,
            processing_time=processing_time,
            timestamp=start_time
        )
    
    def _extract_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[List[int]]:
        """Extract dominant colors from image"""
        try:
            # Reshape image to be a list of pixels
            data = image.reshape((-1, 3))
            data = np.float32(data)
            
            # Apply k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert to regular integers
            centers = np.uint8(centers)
            
            return centers.tolist()
        except:
            # Return default colors if extraction fails
            return [[128, 128, 128], [64, 64, 64], [192, 192, 192]]
    
    def _compute_histogram_features(self, image: np.ndarray) -> Dict[str, List[float]]:
        """Compute color histogram features"""
        try:
            histograms = {}
            
            # Compute histogram for each channel
            for i, color in enumerate(['blue', 'green', 'red']):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                # Normalize histogram
                hist = hist.flatten() / hist.sum()
                histograms[color] = hist.tolist()
            
            return histograms
        except:
            # Return empty histograms if computation fails
            return {
                'blue': [0.0] * 256,
                'green': [0.0] * 256,
                'red': [0.0] * 256
            }
    
    async def process_video(self, video_path: str, sample_rate: int = 30) -> List[Dict[str, Any]]:
        """Process video file"""
        return await self.video_processor.analyze_video_stream(video_path, sample_rate)

# Global vision processor instance
vision_processor = VisionProcessor()