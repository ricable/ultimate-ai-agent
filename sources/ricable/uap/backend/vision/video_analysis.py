"""
Video Analysis Module

Provides comprehensive video analysis capabilities including:
- Real-time video stream processing
- Object tracking and motion detection
- Action recognition and activity analysis
- Video summarization and key frame extraction
- Multi-object tracking across frames
- Video quality assessment
- Scene change detection
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
from pathlib import Path
import numpy as np
import json
import time
from collections import deque
import threading
import queue

try:
    import cv2
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    from transformers import pipeline
    import imageio
    import av
    from ultralytics import YOLO
    import mediapipe as mp
    HAS_VIDEO_DEPS = True
except ImportError:
    HAS_VIDEO_DEPS = False
    cv2 = None
    Image = None
    torch = None
    imageio = None
    av = None
    YOLO = None
    mp = None

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    """Advanced video analysis with AI-powered capabilities."""
    
    def __init__(self, model_cache_dir: Optional[str] = None):
        """Initialize video analyzer with AI models."""
        self.model_cache_dir = model_cache_dir or "./models/video"
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.trackers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not HAS_VIDEO_DEPS:
            logger.warning("Video dependencies not installed. Limited functionality available.")
            return
            
        # Initialize models and trackers
        self._initialize_models()
        self._initialize_trackers()
    
    def _initialize_models(self):
        """Initialize video analysis models."""
        try:
            # Object detection for video
            self.models['yolo'] = YOLO('yolov8n.pt')
            
            # Action recognition pipeline
            self.models['action_recognition'] = pipeline(
                "video-classification",
                model="microsoft/videomae-base-finetuned-kinetics"
            )
            
            # MediaPipe for pose detection
            self.models['pose'] = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # MediaPipe for face detection
            self.models['face'] = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            
            # MediaPipe for hand tracking
            self.models['hands'] = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logger.info("Video analysis models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing video models: {e}")
            self.models = {}
    
    def _initialize_trackers(self):
        """Initialize object trackers."""
        try:
            # Available OpenCV trackers
            self.tracker_types = [
                'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT'
            ]
            
            logger.info("Video trackers initialized")
            
        except Exception as e:
            logger.error(f"Error initializing trackers: {e}")
    
    async def analyze_video(self, video_path: str, 
                          analysis_types: List[str],
                          sample_rate: int = 1) -> Dict[str, Any]:
        """
        Comprehensive video analysis.
        
        Args:
            video_path: Path to video file or camera index
            analysis_types: Types of analysis to perform
            sample_rate: Process every N frames (1 = every frame)
        
        Returns:
            Dictionary with analysis results
        """
        if not HAS_VIDEO_DEPS:
            return {"error": "Video dependencies not installed"}
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": f"Could not open video: {video_path}"}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            results = {
                "video_info": {
                    "fps": fps,
                    "frame_count": frame_count,
                    "resolution": [width, height],
                    "duration": duration
                },
                "analysis_types": analysis_types,
                "results": {}
            }
            
            # Process frames
            frame_results = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames based on sample_rate
                if frame_idx % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Perform requested analyses
                    frame_analysis = await self._analyze_frame(
                        frame_rgb, frame_idx, analysis_types
                    )
                    frame_analysis["timestamp"] = frame_idx / fps
                    frame_results.append(frame_analysis)
                
                frame_idx += 1
            
            cap.release()
            
            # Aggregate results
            results["frame_results"] = frame_results
            results["summary"] = await self._aggregate_video_results(frame_results, results["video_info"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            return {"error": str(e)}
    
    async def track_objects_in_video(self, video_path: str,
                                   object_classes: List[str] = None,
                                   tracker_type: str = "CSRT") -> Dict[str, Any]:
        """
        Track objects throughout a video.
        
        Args:
            video_path: Path to video file
            object_classes: Classes to track (None for all)
            tracker_type: Type of tracker to use
        
        Returns:
            Dictionary with tracking results
        """
        if not HAS_VIDEO_DEPS:
            return {"error": "Video tracking not available"}
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": f"Could not open video: {video_path}"}
            
            # Get first frame for initial detection
            ret, first_frame = cap.read()
            if not ret:
                return {"error": "Could not read first frame"}
            
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            
            # Detect objects in first frame
            if 'yolo' in self.models:
                detections = self.models['yolo'](first_frame_rgb)
                initial_objects = []
                
                for result in detections:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_name = result.names[int(box.cls[0])]
                            if object_classes is None or class_name in object_classes:
                                bbox = box.xyxy[0].tolist()
                                initial_objects.append({
                                    "class": class_name,
                                    "bbox": bbox,
                                    "confidence": float(box.conf[0]),
                                    "id": len(initial_objects)
                                })
            else:
                return {"error": "Object detection model not available"}
            
            if not initial_objects:
                return {"error": "No objects detected in first frame"}
            
            # Initialize trackers for each object
            trackers = []
            for obj in initial_objects:
                tracker = self._create_tracker(tracker_type)
                if tracker:
                    bbox = obj["bbox"]
                    # Convert to (x, y, width, height) format
                    init_bbox = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                    tracker.init(first_frame, init_bbox)
                    trackers.append({
                        "tracker": tracker,
                        "object_id": obj["id"],
                        "class": obj["class"],
                        "active": True
                    })
            
            # Track objects through video
            tracking_results = []
            frame_idx = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_tracks = {
                    "frame": frame_idx,
                    "timestamp": frame_idx / fps if fps > 0 else 0,
                    "objects": []
                }
                
                # Update trackers
                for tracker_info in trackers:
                    if tracker_info["active"]:
                        success, bbox = tracker_info["tracker"].update(frame)
                        
                        if success:
                            # Convert back to (x1, y1, x2, y2) format
                            x, y, w, h = bbox
                            bbox_coords = [x, y, x + w, y + h]
                            
                            frame_tracks["objects"].append({
                                "object_id": tracker_info["object_id"],
                                "class": tracker_info["class"],
                                "bbox": bbox_coords,
                                "confidence": 1.0  # Tracker doesn't provide confidence
                            })
                        else:
                            # Tracker lost object
                            tracker_info["active"] = False
                
                tracking_results.append(frame_tracks)
                frame_idx += 1
            
            cap.release()
            
            # Generate tracking summary
            summary = await self._generate_tracking_summary(tracking_results, initial_objects)
            
            return {
                "initial_objects": initial_objects,
                "tracking_results": tracking_results,
                "summary": summary,
                "tracker_type": tracker_type
            }
            
        except Exception as e:
            logger.error(f"Error in object tracking: {e}")
            return {"error": str(e)}
    
    async def detect_scene_changes(self, video_path: str,
                                 threshold: float = 0.3) -> Dict[str, Any]:
        """
        Detect scene changes in video.
        
        Args:
            video_path: Path to video file
            threshold: Threshold for scene change detection
        
        Returns:
            Dictionary with scene change results
        """
        if not HAS_VIDEO_DEPS:
            return {"error": "Scene change detection not available"}
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": f"Could not open video: {video_path}"}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            scene_changes = []
            prev_hist = None
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to HSV for better histogram comparison
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Calculate histogram
                hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
                
                if prev_hist is not None:
                    # Compare histograms
                    correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    
                    # Scene change detected if correlation is below threshold
                    if correlation < threshold:
                        scene_changes.append({
                            "frame": frame_idx,
                            "timestamp": frame_idx / fps if fps > 0 else 0,
                            "correlation": correlation,
                            "change_magnitude": 1 - correlation
                        })
                
                prev_hist = hist
                frame_idx += 1
            
            cap.release()
            
            return {
                "scene_changes": scene_changes,
                "total_scenes": len(scene_changes) + 1,
                "threshold_used": threshold
            }
            
        except Exception as e:
            logger.error(f"Error detecting scene changes: {e}")
            return {"error": str(e)}
    
    async def extract_key_frames(self, video_path: str,
                               num_frames: int = 10) -> Dict[str, Any]:
        """
        Extract key frames from video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of key frames to extract
        
        Returns:
            Dictionary with key frame information
        """
        if not HAS_VIDEO_DEPS:
            return {"error": "Key frame extraction not available"}
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": f"Could not open video: {video_path}"}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame indices to extract
            if num_frames >= frame_count:
                frame_indices = list(range(frame_count))
            else:
                step = frame_count // num_frames
                frame_indices = [i * step for i in range(num_frames)]
            
            key_frames = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to base64 for storage/transmission
                    pil_image = Image.fromarray(frame_rgb)
                    import io
                    import base64
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    
                    key_frames.append({
                        "frame_index": frame_idx,
                        "timestamp": frame_idx / fps if fps > 0 else 0,
                        "image_data": f"data:image/jpeg;base64,{img_str}",
                        "resolution": [frame.shape[1], frame.shape[0]]
                    })
            
            cap.release()
            
            return {
                "key_frames": key_frames,
                "total_frames_extracted": len(key_frames),
                "extraction_method": "uniform_sampling"
            }
            
        except Exception as e:
            logger.error(f"Error extracting key frames: {e}")
            return {"error": str(e)}
    
    def start_live_stream_analysis(self, source: Union[str, int],
                                 analysis_types: List[str],
                                 callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Start live video stream analysis.
        
        Args:
            source: Video source (camera index or stream URL)
            analysis_types: Types of analysis to perform
            callback: Optional callback function for results
        
        Returns:
            Dictionary with stream info
        """
        if not HAS_VIDEO_DEPS:
            return {"error": "Live stream analysis not available"}
        
        try:
            # Create a unique stream ID
            stream_id = f"stream_{int(time.time())}"
            
            # Start streaming thread
            stream_thread = threading.Thread(
                target=self._live_stream_worker,
                args=(source, analysis_types, callback, stream_id)
            )
            stream_thread.daemon = True
            stream_thread.start()
            
            return {
                "stream_id": stream_id,
                "status": "started",
                "analysis_types": analysis_types
            }
            
        except Exception as e:
            logger.error(f"Error starting live stream: {e}")
            return {"error": str(e)}
    
    def _live_stream_worker(self, source: Union[str, int],
                          analysis_types: List[str],
                          callback: Optional[callable],
                          stream_id: str):
        """Worker function for live stream analysis."""
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.error(f"Could not open video source: {source}")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_time = 1.0 / fps
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Convert frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Analyze frame
                try:
                    analysis = asyncio.run(self._analyze_frame(frame_rgb, 0, analysis_types))
                    analysis["stream_id"] = stream_id
                    analysis["timestamp"] = time.time()
                    
                    # Call callback if provided
                    if callback:
                        callback(analysis)
                        
                except Exception as e:
                    logger.error(f"Error in frame analysis: {e}")
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error in live stream worker: {e}")
    
    async def _analyze_frame(self, frame: np.ndarray, frame_idx: int,
                           analysis_types: List[str]) -> Dict[str, Any]:
        """Analyze a single frame."""
        analysis = {
            "frame_index": frame_idx,
            "frame_size": frame.shape[:2]
        }
        
        try:
            if "object_detection" in analysis_types and 'yolo' in self.models:
                analysis["objects"] = await self._detect_objects_in_frame(frame)
            
            if "pose_detection" in analysis_types and 'pose' in self.models:
                analysis["poses"] = await self._detect_poses_in_frame(frame)
            
            if "face_detection" in analysis_types and 'face' in self.models:
                analysis["faces"] = await self._detect_faces_in_frame(frame)
            
            if "hand_tracking" in analysis_types and 'hands' in self.models:
                analysis["hands"] = await self._detect_hands_in_frame(frame)
            
            if "motion_detection" in analysis_types:
                analysis["motion"] = await self._detect_motion_in_frame(frame, frame_idx)
            
            if "quality_assessment" in analysis_types:
                analysis["quality"] = await self._assess_frame_quality(frame)
            
        except Exception as e:
            logger.error(f"Error analyzing frame {frame_idx}: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    async def _detect_objects_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in a single frame."""
        try:
            results = self.models['yolo'](frame)
            objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        objects.append({
                            "class": result.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": box.xyxy[0].tolist()
                        })
            
            return objects
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return []
    
    async def _detect_poses_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """Detect poses in a single frame."""
        try:
            results = self.models['pose'].process(frame)
            poses = []
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })
                
                poses.append({
                    "landmarks": landmarks,
                    "confidence": 0.8  # MediaPipe doesn't provide overall confidence
                })
            
            return poses
            
        except Exception as e:
            logger.error(f"Pose detection error: {e}")
            return []
    
    async def _detect_faces_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in a single frame."""
        try:
            results = self.models['face'].process(frame)
            faces = []
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    faces.append({
                        "bbox": [
                            bbox.xmin * frame.shape[1],
                            bbox.ymin * frame.shape[0],
                            (bbox.xmin + bbox.width) * frame.shape[1],
                            (bbox.ymin + bbox.height) * frame.shape[0]
                        ],
                        "confidence": detection.score[0]
                    })
            
            return faces
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    async def _detect_hands_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """Detect hands in a single frame."""
        try:
            results = self.models['hands'].process(frame)
            hands = []
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append({
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z
                        })
                    
                    hands.append({
                        "landmarks": landmarks,
                        "confidence": 0.8  # MediaPipe doesn't provide confidence
                    })
            
            return hands
            
        except Exception as e:
            logger.error(f"Hand detection error: {e}")
            return []
    
    async def _detect_motion_in_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
        """Detect motion in frame (simplified implementation)."""
        # This is a placeholder - real motion detection would require frame history
        return {
            "motion_detected": False,
            "motion_magnitude": 0.0,
            "motion_regions": []
        }
    
    async def _assess_frame_quality(self, frame: np.ndarray) -> Dict:
        """Assess frame quality."""
        try:
            # Convert to grayscale for quality assessment
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            
            # Laplacian variance (focus measure)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Brightness
            brightness = np.mean(gray)
            
            # Contrast
            contrast = np.std(gray)
            
            return {
                "sharpness": float(laplacian_var),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "overall_quality": min(100, laplacian_var / 1000 * 100)
            }
            
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return {"sharpness": 0, "brightness": 0, "contrast": 0, "overall_quality": 0}
    
    async def _aggregate_video_results(self, frame_results: List[Dict],
                                     video_info: Dict) -> Dict[str, Any]:
        """Aggregate frame results into video summary."""
        try:
            summary = {
                "total_frames_processed": len(frame_results),
                "processing_rate": len(frame_results) / video_info["duration"] if video_info["duration"] > 0 else 0
            }
            
            # Aggregate object detection results
            if any("objects" in frame for frame in frame_results):
                all_objects = []
                for frame in frame_results:
                    if "objects" in frame:
                        all_objects.extend(frame["objects"])
                
                if all_objects:
                    object_classes = [obj["class"] for obj in all_objects]
                    summary["object_detection"] = {
                        "total_detections": len(all_objects),
                        "unique_classes": len(set(object_classes)),
                        "class_counts": {cls: object_classes.count(cls) for cls in set(object_classes)},
                        "avg_confidence": np.mean([obj["confidence"] for obj in all_objects])
                    }
            
            # Aggregate pose and face detection
            pose_frames = sum(1 for frame in frame_results if frame.get("poses"))
            face_frames = sum(1 for frame in frame_results if frame.get("faces"))
            
            summary["human_detection"] = {
                "frames_with_poses": pose_frames,
                "frames_with_faces": face_frames,
                "pose_detection_rate": pose_frames / len(frame_results) if frame_results else 0,
                "face_detection_rate": face_frames / len(frame_results) if frame_results else 0
            }
            
            # Quality assessment
            quality_scores = [frame.get("quality", {}).get("overall_quality", 0) for frame in frame_results]
            if quality_scores:
                summary["quality_assessment"] = {
                    "average_quality": np.mean(quality_scores),
                    "min_quality": np.min(quality_scores),
                    "max_quality": np.max(quality_scores),
                    "quality_consistency": 1 - (np.std(quality_scores) / 100)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error aggregating results: {e}")
            return {"error": str(e)}
    
    async def _generate_tracking_summary(self, tracking_results: List[Dict],
                                       initial_objects: List[Dict]) -> Dict[str, Any]:
        """Generate tracking summary."""
        try:
            summary = {
                "initial_objects": len(initial_objects),
                "frames_processed": len(tracking_results),
                "object_statistics": {}
            }
            
            # Analyze each object's tracking performance
            for obj in initial_objects:
                obj_id = obj["id"]
                obj_class = obj["class"]
                
                # Count frames where object was tracked
                tracked_frames = 0
                for frame in tracking_results:
                    if any(o["object_id"] == obj_id for o in frame["objects"]):
                        tracked_frames += 1
                
                tracking_rate = tracked_frames / len(tracking_results) if tracking_results else 0
                
                summary["object_statistics"][f"{obj_class}_{obj_id}"] = {
                    "class": obj_class,
                    "tracked_frames": tracked_frames,
                    "tracking_rate": tracking_rate,
                    "tracking_quality": "good" if tracking_rate > 0.8 else "fair" if tracking_rate > 0.5 else "poor"
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating tracking summary: {e}")
            return {"error": str(e)}
    
    def _create_tracker(self, tracker_type: str):
        """Create OpenCV tracker of specified type."""
        try:
            if tracker_type == "BOOSTING":
                return cv2.legacy.TrackerBoosting_create()
            elif tracker_type == "MIL":
                return cv2.legacy.TrackerMIL_create()
            elif tracker_type == "KCF":
                return cv2.legacy.TrackerKCF_create()
            elif tracker_type == "TLD":
                return cv2.legacy.TrackerTLD_create()
            elif tracker_type == "MEDIANFLOW":
                return cv2.legacy.TrackerMedianFlow_create()
            elif tracker_type == "MOSSE":
                return cv2.legacy.TrackerMOSSE_create()
            elif tracker_type == "CSRT":
                return cv2.TrackerCSRT_create()
            else:
                logger.warning(f"Unknown tracker type: {tracker_type}, using CSRT")
                return cv2.TrackerCSRT_create()
                
        except Exception as e:
            logger.error(f"Error creating tracker: {e}")
            return None
    
    def get_supported_analysis_types(self) -> List[str]:
        """Get list of supported video analysis types."""
        return [
            "object_detection",
            "pose_detection", 
            "face_detection",
            "hand_tracking",
            "motion_detection",
            "quality_assessment"
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "models_loaded": list(self.models.keys()),
            "trackers_available": self.tracker_types,
            "device": self.device,
            "video_deps_available": HAS_VIDEO_DEPS,
            "supported_analysis_types": self.get_supported_analysis_types()
        }