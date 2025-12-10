"""
Computer Vision and Multimodal AI Package

This package provides comprehensive computer vision capabilities including:
- Image and video processing
- Enhanced OCR with layout understanding
- Visual question answering and image generation
- Document layout analysis and table extraction
- Real-time video analysis and streaming
"""

from .image_processing import ImageProcessor
from .ocr_enhanced import EnhancedOCR
from .visual_qa import VisualQA
from .video_analysis import VideoAnalyzer
from .document_layout import DocumentLayoutAnalyzer

__all__ = [
    "ImageProcessor",
    "EnhancedOCR", 
    "VisualQA",
    "VideoAnalyzer",
    "DocumentLayoutAnalyzer",
]

__version__ = "1.0.0"