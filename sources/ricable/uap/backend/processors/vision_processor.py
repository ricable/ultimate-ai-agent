"""
Vision Processor

Central processor for all computer vision and multimodal AI operations.
Coordinates between different vision modules and provides unified interface.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import time

from ..vision import (
    ImageProcessor,
    EnhancedOCR,
    VisualQA,
    VideoAnalyzer,
    DocumentLayoutAnalyzer
)

logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Central vision processor that coordinates all vision capabilities.
    
    Provides unified interface for:
    - Image processing and enhancement
    - OCR with layout understanding
    - Visual question answering
    - Video analysis and streaming
    - Document layout analysis
    """
    
    def __init__(self, model_cache_dir: Optional[str] = None):
        """Initialize vision processor with all sub-modules."""
        self.model_cache_dir = model_cache_dir or "./models/vision"
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize all vision modules
        self.image_processor = ImageProcessor(model_cache_dir)
        self.ocr_processor = EnhancedOCR(model_cache_dir)
        self.vqa_processor = VisualQA(model_cache_dir)
        self.video_processor = VideoAnalyzer(model_cache_dir)
        self.layout_processor = DocumentLayoutAnalyzer(model_cache_dir)
        
        # Cache for processed results
        self.result_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        logger.info("Vision processor initialized with all modules")
    
    async def process_vision_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a vision request with automatic routing to appropriate modules.
        
        Args:
            request: Vision processing request containing:
                - operation: Type of operation
                - data: Image/video data
                - parameters: Additional parameters
        
        Returns:
            Dictionary with processing results
        """
        try:
            operation = request.get("operation")
            data = request.get("data")
            parameters = request.get("parameters", {})
            
            if not operation or not data:
                return {"error": "Missing operation or data in request"}
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Route to appropriate processor
            start_time = time.time()
            result = await self._route_request(operation, data, parameters)
            processing_time = time.time() - start_time
            
            # Add metadata
            result["processing_metadata"] = {
                "operation": operation,
                "processing_time": processing_time,
                "timestamp": time.time(),
                "processor": "vision_processor"
            }
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing vision request: {e}")
            return {"error": str(e)}
    
    async def _route_request(self, operation: str, data: Any, parameters: Dict) -> Dict[str, Any]:
        """Route request to appropriate vision module."""
        
        if operation.startswith("image_"):
            return await self._handle_image_operation(operation, data, parameters)
        elif operation.startswith("ocr_"):
            return await self._handle_ocr_operation(operation, data, parameters)
        elif operation.startswith("vqa_"):
            return await self._handle_vqa_operation(operation, data, parameters)
        elif operation.startswith("video_"):
            return await self._handle_video_operation(operation, data, parameters)
        elif operation.startswith("layout_"):
            return await self._handle_layout_operation(operation, data, parameters)
        elif operation in ["analyze_image", "analyze_document", "analyze_video"]:
            return await self._handle_comprehensive_analysis(operation, data, parameters)
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    async def _handle_image_operation(self, operation: str, data: Any, parameters: Dict) -> Dict[str, Any]:
        """Handle image processing operations."""
        try:
            if operation == "image_process":
                operations = parameters.get("operations", ["enhance"])
                return await self.image_processor.process_image(data, operations)
            
            elif operation == "image_enhance":
                result = await self.image_processor.process_image(data, ["enhance"])
                return result.get("results", {}).get("enhance", {})
            
            elif operation == "image_detect_objects":
                result = await self.image_processor.process_image(data, ["detect_objects"])
                return result.get("results", {}).get("detect_objects", {})
            
            elif operation == "image_classify":
                result = await self.image_processor.process_image(data, ["classify"])
                return result.get("results", {}).get("classify", {})
            
            elif operation == "image_analyze_colors":
                result = await self.image_processor.process_image(data, ["analyze_colors"])
                return result.get("results", {}).get("analyze_colors", {})
            
            elif operation == "image_batch_process":
                images = parameters.get("images", [])
                operations = parameters.get("operations", ["enhance"])
                return await self.image_processor.batch_process(images, operations)
            
            else:
                return {"error": f"Unknown image operation: {operation}"}
                
        except Exception as e:
            return {"error": f"Image processing error: {e}"}
    
    async def _handle_ocr_operation(self, operation: str, data: Any, parameters: Dict) -> Dict[str, Any]:
        """Handle OCR operations."""
        try:
            if operation == "ocr_extract_text":
                engines = parameters.get("engines", None)
                enhance_image = parameters.get("enhance_image", True)
                detect_language = parameters.get("detect_language", True)
                return await self.ocr_processor.extract_text(data, engines, enhance_image, detect_language)
            
            elif operation == "ocr_analyze_layout":
                include_reading_order = parameters.get("include_reading_order", True)
                return await self.ocr_processor.analyze_layout(data, include_reading_order)
            
            elif operation == "ocr_extract_tables":
                return await self.ocr_processor.extract_tables(data)
            
            else:
                return {"error": f"Unknown OCR operation: {operation}"}
                
        except Exception as e:
            return {"error": f"OCR processing error: {e}"}
    
    async def _handle_vqa_operation(self, operation: str, data: Any, parameters: Dict) -> Dict[str, Any]:
        """Handle visual question answering operations."""
        try:
            if operation == "vqa_answer_question":
                question = parameters.get("question", "")
                model = parameters.get("model", "blip")
                return await self.vqa_processor.answer_question(data, question, model)
            
            elif operation == "vqa_generate_caption":
                max_length = parameters.get("max_length", 50)
                num_captions = parameters.get("num_captions", 1)
                return await self.vqa_processor.generate_caption(data, max_length, num_captions)
            
            elif operation == "vqa_describe_scene":
                return await self.vqa_processor.describe_scene(data)
            
            elif operation == "vqa_analyze_spatial":
                return await self.vqa_processor.analyze_spatial_relationships(data)
            
            elif operation == "vqa_compare_images":
                image2 = parameters.get("image2")
                if not image2:
                    return {"error": "Second image required for comparison"}
                return await self.vqa_processor.compare_images(data, image2)
            
            elif operation == "vqa_batch_questions":
                questions = parameters.get("questions", [])
                model = parameters.get("model", "blip")
                return await self.vqa_processor.batch_vqa(data, questions, model)
            
            else:
                return {"error": f"Unknown VQA operation: {operation}"}
                
        except Exception as e:
            return {"error": f"VQA processing error: {e}"}
    
    async def _handle_video_operation(self, operation: str, data: Any, parameters: Dict) -> Dict[str, Any]:
        """Handle video analysis operations."""
        try:
            if operation == "video_analyze":
                analysis_types = parameters.get("analysis_types", ["object_detection"])
                sample_rate = parameters.get("sample_rate", 1)
                return await self.video_processor.analyze_video(data, analysis_types, sample_rate)
            
            elif operation == "video_track_objects":
                object_classes = parameters.get("object_classes", None)
                tracker_type = parameters.get("tracker_type", "CSRT")
                return await self.video_processor.track_objects_in_video(data, object_classes, tracker_type)
            
            elif operation == "video_detect_scene_changes":
                threshold = parameters.get("threshold", 0.3)
                return await self.video_processor.detect_scene_changes(data, threshold)
            
            elif operation == "video_extract_key_frames":
                num_frames = parameters.get("num_frames", 10)
                return await self.video_processor.extract_key_frames(data, num_frames)
            
            elif operation == "video_start_live_stream":
                analysis_types = parameters.get("analysis_types", ["object_detection"])
                callback = parameters.get("callback", None)
                return self.video_processor.start_live_stream_analysis(data, analysis_types, callback)
            
            else:
                return {"error": f"Unknown video operation: {operation}"}
                
        except Exception as e:
            return {"error": f"Video processing error: {e}"}
    
    async def _handle_layout_operation(self, operation: str, data: Any, parameters: Dict) -> Dict[str, Any]:
        """Handle document layout operations."""
        try:
            if operation == "layout_analyze_document":
                document_type = parameters.get("document_type", "auto")
                return await self.layout_processor.analyze_document_layout(data, document_type)
            
            elif operation == "layout_extract_text_regions":
                classify_regions = parameters.get("classify_regions", True)
                return await self.layout_processor.extract_text_regions(data, classify_regions)
            
            elif operation == "layout_detect_tables":
                return await self.layout_processor.detect_tables(data)
            
            elif operation == "layout_analyze_forms":
                return await self.layout_processor.analyze_form_fields(data)
            
            else:
                return {"error": f"Unknown layout operation: {operation}"}
                
        except Exception as e:
            return {"error": f"Layout processing error: {e}"}
    
    async def _handle_comprehensive_analysis(self, operation: str, data: Any, parameters: Dict) -> Dict[str, Any]:
        """Handle comprehensive analysis operations that use multiple modules."""
        try:
            if operation == "analyze_image":
                return await self._comprehensive_image_analysis(data, parameters)
            elif operation == "analyze_document":
                return await self._comprehensive_document_analysis(data, parameters)
            elif operation == "analyze_video":
                return await self._comprehensive_video_analysis(data, parameters)
            else:
                return {"error": f"Unknown comprehensive operation: {operation}"}
                
        except Exception as e:
            return {"error": f"Comprehensive analysis error: {e}"}
    
    async def _comprehensive_image_analysis(self, data: Any, parameters: Dict) -> Dict[str, Any]:
        """Perform comprehensive image analysis using multiple modules."""
        try:
            results = {
                "analysis_type": "comprehensive_image",
                "modules_used": []
            }
            
            # Basic image processing
            try:
                image_result = await self.image_processor.process_image(
                    data, ["enhance", "detect_objects", "classify", "analyze_colors"]
                )
                results["image_processing"] = image_result
                results["modules_used"].append("image_processor")
            except Exception as e:
                logger.warning(f"Image processing failed: {e}")
            
            # Visual QA - generate caption and answer common questions
            try:
                caption_result = await self.vqa_processor.generate_caption(data)
                scene_result = await self.vqa_processor.describe_scene(data)
                
                results["visual_analysis"] = {
                    "caption": caption_result,
                    "scene_description": scene_result
                }
                results["modules_used"].append("vqa_processor")
            except Exception as e:
                logger.warning(f"VQA processing failed: {e}")
            
            # OCR if text might be present
            try:
                ocr_result = await self.ocr_processor.extract_text(data, enhance_image=True)
                if ocr_result.get("combined_text", {}).get("word_count", 0) > 0:
                    results["text_extraction"] = ocr_result
                    results["modules_used"].append("ocr_processor")
            except Exception as e:
                logger.warning(f"OCR processing failed: {e}")
            
            return results
            
        except Exception as e:
            return {"error": f"Comprehensive image analysis failed: {e}"}
    
    async def _comprehensive_document_analysis(self, data: Any, parameters: Dict) -> Dict[str, Any]:
        """Perform comprehensive document analysis."""
        try:
            results = {
                "analysis_type": "comprehensive_document",
                "modules_used": []
            }
            
            # Document layout analysis
            try:
                layout_result = await self.layout_processor.analyze_document_layout(data)
                results["layout_analysis"] = layout_result
                results["modules_used"].append("layout_processor")
            except Exception as e:
                logger.warning(f"Layout analysis failed: {e}")
            
            # Enhanced OCR
            try:
                ocr_result = await self.ocr_processor.extract_text(data, enhance_image=True, detect_language=True)
                layout_ocr = await self.ocr_processor.analyze_layout(data)
                
                results["text_extraction"] = {
                    "ocr_results": ocr_result,
                    "layout_ocr": layout_ocr
                }
                results["modules_used"].append("ocr_processor")
            except Exception as e:
                logger.warning(f"OCR processing failed: {e}")
            
            # Table extraction if tables detected
            try:
                table_result = await self.layout_processor.detect_tables(data)
                if table_result.get("tables_detected", 0) > 0:
                    detailed_tables = await self.ocr_processor.extract_tables(data)
                    results["table_analysis"] = {
                        "layout_tables": table_result,
                        "extracted_tables": detailed_tables
                    }
            except Exception as e:
                logger.warning(f"Table analysis failed: {e}")
            
            # Form analysis
            try:
                form_result = await self.layout_processor.analyze_form_fields(data)
                if form_result.get("total_fields", 0) > 0:
                    results["form_analysis"] = form_result
            except Exception as e:
                logger.warning(f"Form analysis failed: {e}")
            
            return results
            
        except Exception as e:
            return {"error": f"Comprehensive document analysis failed: {e}"}
    
    async def _comprehensive_video_analysis(self, data: Any, parameters: Dict) -> Dict[str, Any]:
        """Perform comprehensive video analysis."""
        try:
            results = {
                "analysis_type": "comprehensive_video",
                "modules_used": ["video_processor"]
            }
            
            # Basic video analysis
            analysis_types = [
                "object_detection", "pose_detection", "face_detection", 
                "quality_assessment"
            ]
            sample_rate = parameters.get("sample_rate", 5)  # Every 5th frame
            
            video_result = await self.video_processor.analyze_video(data, analysis_types, sample_rate)
            results["video_analysis"] = video_result
            
            # Scene change detection
            try:
                scene_changes = await self.video_processor.detect_scene_changes(data)
                results["scene_analysis"] = scene_changes
            except Exception as e:
                logger.warning(f"Scene change detection failed: {e}")
            
            # Key frame extraction
            try:
                num_frames = parameters.get("key_frames", 10)
                key_frames = await self.video_processor.extract_key_frames(data, num_frames)
                results["key_frames"] = key_frames
            except Exception as e:
                logger.warning(f"Key frame extraction failed: {e}")
            
            return results
            
        except Exception as e:
            return {"error": f"Comprehensive video analysis failed: {e}"}
    
    def _generate_cache_key(self, request: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        try:
            # Create a hash-like key from request parameters
            operation = request.get("operation", "")
            parameters = request.get("parameters", {})
            
            # Don't include actual data in cache key, just metadata
            key_parts = [
                operation,
                str(sorted(parameters.items())) if parameters else ""
            ]
            
            return "_".join(key_parts)
            
        except Exception:
            return f"cache_{int(time.time())}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if valid."""
        try:
            if cache_key in self.result_cache:
                cached_data = self.result_cache[cache_key]
                if time.time() - cached_data["timestamp"] < self.cache_ttl:
                    logger.info(f"Returning cached result for {cache_key}")
                    return cached_data["result"]
                else:
                    # Remove expired cache
                    del self.result_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
            return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache processing result."""
        try:
            self.result_cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
            
            # Limit cache size
            if len(self.result_cache) > 100:
                # Remove oldest entries
                sorted_items = sorted(
                    self.result_cache.items(),
                    key=lambda x: x[1]["timestamp"]
                )
                for key, _ in sorted_items[:50]:  # Remove half
                    del self.result_cache[key]
                    
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get vision processor capabilities."""
        return {
            "modules": {
                "image_processor": self.image_processor.get_supported_operations(),
                "ocr_processor": self.ocr_processor.get_supported_languages(),
                "vqa_processor": self.vqa_processor.get_supported_models(),
                "video_processor": self.video_processor.get_supported_analysis_types(),
                "layout_processor": self.layout_processor.get_supported_document_types()
            },
            "comprehensive_operations": [
                "analyze_image",
                "analyze_document", 
                "analyze_video"
            ],
            "cache_enabled": True,
            "cache_ttl": self.cache_ttl
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for all vision modules."""
        return {
            "image_processor": self.image_processor.get_model_info(),
            "ocr_processor": self.ocr_processor.get_engine_info(),
            "vqa_processor": self.vqa_processor.get_model_info(),
            "video_processor": self.video_processor.get_model_info(),
            "layout_processor": self.layout_processor.get_model_info(),
            "cache_stats": {
                "cached_items": len(self.result_cache),
                "cache_ttl": self.cache_ttl
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all vision modules."""
        health_status = {
            "overall_status": "healthy",
            "modules": {},
            "timestamp": time.time()
        }
        
        # Check each module
        modules = [
            ("image_processor", self.image_processor),
            ("ocr_processor", self.ocr_processor),
            ("vqa_processor", self.vqa_processor),
            ("video_processor", self.video_processor),
            ("layout_processor", self.layout_processor)
        ]
        
        for module_name, module in modules:
            try:
                if hasattr(module, 'get_model_info'):
                    info = module.get_model_info()
                    health_status["modules"][module_name] = {
                        "status": "healthy",
                        "models_loaded": len(info.get("models_loaded", [])),
                        "deps_available": info.get("vision_deps_available", True) or 
                                       info.get("ocr_deps_available", True) or
                                       info.get("vqa_deps_available", True) or
                                       info.get("video_deps_available", True) or
                                       info.get("layout_deps_available", True)
                    }
                else:
                    health_status["modules"][module_name] = {"status": "unknown"}
                    
            except Exception as e:
                health_status["modules"][module_name] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
        
        return health_status