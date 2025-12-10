"""
Visual Question Answering (VQA) Module

Provides advanced visual question answering capabilities including:
- Image-based question answering using multimodal models
- Scene understanding and description
- Object relationships and spatial reasoning
- Text-to-image generation
- Image captioning and description
- Visual reasoning and inference
"""

import asyncio
import base64
import io
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
import json

try:
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    import torch
    import torchvision.transforms as transforms
    from transformers import (
        BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration,
        ViltProcessor, ViltForQuestionAnswering,
        AutoProcessor, AutoModelForCausalLM,
        pipeline
    )
    import clip
    from sentence_transformers import SentenceTransformer
    import requests
    HAS_VQA_DEPS = True
except ImportError:
    HAS_VQA_DEPS = False
    cv2 = None
    Image = None
    torch = None

logger = logging.getLogger(__name__)

class VisualQA:
    """Advanced Visual Question Answering with multimodal AI capabilities."""
    
    def __init__(self, model_cache_dir: Optional[str] = None):
        """Initialize VQA with multimodal models."""
        self.model_cache_dir = model_cache_dir or "./models/vqa"
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.processors = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not HAS_VQA_DEPS:
            logger.warning("VQA dependencies not installed. Limited functionality available.")
            return
            
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize VQA and multimodal models."""
        try:
            # BLIP for VQA and image captioning
            self.processors['blip_vqa'] = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.models['blip_vqa'] = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            
            self.processors['blip_caption'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.models['blip_caption'] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # ViLT for VQA
            self.processors['vilt'] = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            self.models['vilt'] = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            
            # CLIP for image-text similarity
            self.models['clip'], self.processors['clip'] = clip.load("ViT-B/32", device=self.device)
            
            # Sentence transformer for text embeddings
            self.models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Image classification pipeline
            self.models['image_classifier'] = pipeline("image-classification", 
                                                     model="microsoft/resnet-50")
            
            # Object detection pipeline
            self.models['object_detector'] = pipeline("object-detection",
                                                    model="facebook/detr-resnet-50")
            
            logger.info("VQA models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing VQA models: {e}")
            self.models = {}
            self.processors = {}
    
    async def answer_question(self, image_data: Union[str, bytes, np.ndarray], 
                            question: str, 
                            model: str = "blip") -> Dict[str, Any]:
        """
        Answer a question about an image.
        
        Args:
            image_data: Image data in various formats
            question: Question about the image
            model: Model to use ("blip", "vilt", "clip")
        
        Returns:
            Dictionary with answer and confidence
        """
        if not HAS_VQA_DEPS:
            return {"error": "VQA dependencies not installed"}
        
        try:
            # Load image
            image = await self._load_image(image_data)
            if image is None:
                return {"error": "Failed to load image"}
            
            pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            
            # Process based on selected model
            if model == "blip" and 'blip_vqa' in self.models:
                result = await self._answer_with_blip(pil_image, question)
            elif model == "vilt" and 'vilt' in self.models:
                result = await self._answer_with_vilt(pil_image, question)
            elif model == "clip" and 'clip' in self.models:
                result = await self._answer_with_clip(pil_image, question)
            else:
                result = {"error": f"Model {model} not available"}
            
            # Add metadata
            result.update({
                "question": question,
                "model_used": model,
                "image_size": image.shape[:2] if isinstance(image, np.ndarray) else pil_image.size
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in VQA: {e}")
            return {"error": str(e)}
    
    async def generate_caption(self, image_data: Union[str, bytes, np.ndarray],
                             max_length: int = 50,
                             num_captions: int = 1) -> Dict[str, Any]:
        """
        Generate captions for an image.
        
        Args:
            image_data: Image data
            max_length: Maximum caption length
            num_captions: Number of captions to generate
        
        Returns:
            Dictionary with generated captions
        """
        if not HAS_VQA_DEPS or 'blip_caption' not in self.models:
            return {"error": "Caption generation not available"}
        
        try:
            # Load image
            image = await self._load_image(image_data)
            if image is None:
                return {"error": "Failed to load image"}
            
            pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            
            # Generate captions
            inputs = self.processors['blip_caption'](pil_image, return_tensors="pt")
            
            captions = []
            for i in range(num_captions):
                with torch.no_grad():
                    output = self.models['blip_caption'].generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=5,
                        do_sample=True if num_captions > 1 else False,
                        temperature=0.7 if num_captions > 1 else 1.0
                    )
                
                caption = self.processors['blip_caption'].decode(output[0], skip_special_tokens=True)
                captions.append({
                    "caption": caption,
                    "length": len(caption.split()),
                    "confidence": 0.8  # BLIP doesn't provide confidence, using default
                })
            
            return {
                "captions": captions,
                "best_caption": captions[0]["caption"] if captions else "",
                "model_used": "blip_caption"
            }
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return {"error": str(e)}
    
    async def describe_scene(self, image_data: Union[str, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Provide comprehensive scene description.
        
        Args:
            image_data: Image data
        
        Returns:
            Dictionary with scene analysis
        """
        if not HAS_VQA_DEPS:
            return {"error": "Scene analysis not available"}
        
        try:
            # Load image
            image = await self._load_image(image_data)
            if image is None:
                return {"error": "Failed to load image"}
            
            pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            
            # Generate comprehensive description
            results = {
                "image_size": image.shape[:2] if isinstance(image, np.ndarray) else pil_image.size,
                "analysis": {}
            }
            
            # Basic caption
            caption_result = await self.generate_caption(image_data)
            if "captions" in caption_result:
                results["analysis"]["caption"] = caption_result["captions"][0]["caption"]
            
            # Object detection
            if 'object_detector' in self.models:
                objects = self.models['object_detector'](pil_image)
                results["analysis"]["objects"] = [
                    {
                        "label": obj["label"],
                        "confidence": obj["score"],
                        "bbox": obj["box"]
                    }
                    for obj in objects
                ]
            
            # Scene classification
            if 'image_classifier' in self.models:
                classification = self.models['image_classifier'](pil_image)
                results["analysis"]["scene_type"] = classification[0]["label"]
                results["analysis"]["scene_confidence"] = classification[0]["score"]
            
            # Answer predefined questions about the scene
            scene_questions = [
                "What is the main subject of this image?",
                "Is this indoors or outdoors?",
                "What time of day is it?",
                "What colors are dominant in this image?",
                "Are there people in this image?",
                "What activity is taking place?"
            ]
            
            qa_results = []
            for question in scene_questions:
                answer = await self.answer_question(image_data, question, model="blip")
                if "answer" in answer:
                    qa_results.append({
                        "question": question,
                        "answer": answer["answer"],
                        "confidence": answer.get("confidence", 0)
                    })
            
            results["analysis"]["detailed_qa"] = qa_results
            
            # Generate structured description
            description = await self._generate_structured_description(results["analysis"])
            results["structured_description"] = description
            
            return results
            
        except Exception as e:
            logger.error(f"Error in scene description: {e}")
            return {"error": str(e)}
    
    async def analyze_spatial_relationships(self, image_data: Union[str, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze spatial relationships between objects in the image.
        
        Args:
            image_data: Image data
        
        Returns:
            Dictionary with spatial analysis
        """
        if not HAS_VQA_DEPS:
            return {"error": "Spatial analysis not available"}
        
        try:
            # Load image
            image = await self._load_image(image_data)
            if image is None:
                return {"error": "Failed to load image"}
            
            pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            
            # Detect objects first
            if 'object_detector' not in self.models:
                return {"error": "Object detector not available"}
            
            objects = self.models['object_detector'](pil_image)
            
            if len(objects) < 2:
                return {"relationships": [], "message": "Need at least 2 objects for spatial analysis"}
            
            # Analyze relationships
            relationships = []
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects[i + 1:], i + 1):
                    relationship = await self._analyze_object_relationship(obj1, obj2)
                    if relationship:
                        relationships.append({
                            "object1": obj1["label"],
                            "object2": obj2["label"],
                            "relationship": relationship,
                            "confidence": min(obj1["score"], obj2["score"])
                        })
            
            # Answer spatial questions
            spatial_questions = [
                "What is on the left side of the image?",
                "What is on the right side of the image?",
                "What is in the center of the image?",
                "What is in the background?",
                "What is in the foreground?"
            ]
            
            spatial_qa = []
            for question in spatial_questions:
                answer = await self.answer_question(image_data, question, model="blip")
                if "answer" in answer:
                    spatial_qa.append({
                        "question": question,
                        "answer": answer["answer"]
                    })
            
            return {
                "detected_objects": len(objects),
                "spatial_relationships": relationships,
                "spatial_qa": spatial_qa,
                "objects": [{"label": obj["label"], "bbox": obj["box"]} for obj in objects]
            }
            
        except Exception as e:
            logger.error(f"Error in spatial analysis: {e}")
            return {"error": str(e)}
    
    async def compare_images(self, image1_data: Union[str, bytes, np.ndarray],
                           image2_data: Union[str, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Compare two images and find similarities/differences.
        
        Args:
            image1_data: First image data
            image2_data: Second image data
        
        Returns:
            Dictionary with comparison results
        """
        if not HAS_VQA_DEPS or 'clip' not in self.models:
            return {"error": "Image comparison not available"}
        
        try:
            # Load images
            image1 = await self._load_image(image1_data)
            image2 = await self._load_image(image2_data)
            
            if image1 is None or image2 is None:
                return {"error": "Failed to load one or both images"}
            
            pil_image1 = Image.fromarray(image1) if isinstance(image1, np.ndarray) else image1
            pil_image2 = Image.fromarray(image2) if isinstance(image2, np.ndarray) else image2
            
            # Generate captions for both images
            caption1 = await self.generate_caption(image1_data)
            caption2 = await self.generate_caption(image2_data)
            
            # CLIP similarity
            image1_input = self.processors['clip'](pil_image1).unsqueeze(0).to(self.device)
            image2_input = self.processors['clip'](pil_image2).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image1_features = self.models['clip'].encode_image(image1_input)
                image2_features = self.models['clip'].encode_image(image2_input)
                
                similarity = torch.cosine_similarity(image1_features, image2_features).item()
            
            # Text similarity of captions
            if caption1.get("captions") and caption2.get("captions"):
                text1 = caption1["captions"][0]["caption"]
                text2 = caption2["captions"][0]["caption"]
                
                embeddings = self.models['sentence_transformer'].encode([text1, text2])
                text_similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
            else:
                text_similarity = 0.0
            
            # Ask comparison questions
            comparison_questions = [
                "Are these images similar?",
                "What is different between these images?",
                "Do these images show the same type of scene?",
                "Which image is brighter?"
            ]
            
            # For comparison questions, we'll analyze each image separately
            comparison_analysis = {}
            for question in comparison_questions:
                answer1 = await self.answer_question(image1_data, question.replace("these images", "this image"))
                answer2 = await self.answer_question(image2_data, question.replace("these images", "this image"))
                
                comparison_analysis[question] = {
                    "image1_answer": answer1.get("answer", ""),
                    "image2_answer": answer2.get("answer", "")
                }
            
            return {
                "visual_similarity": similarity,
                "caption_similarity": text_similarity,
                "overall_similarity": (similarity + text_similarity) / 2,
                "captions": {
                    "image1": caption1.get("captions", [{}])[0].get("caption", ""),
                    "image2": caption2.get("captions", [{}])[0].get("caption", "")
                },
                "comparison_analysis": comparison_analysis,
                "similarity_level": self._get_similarity_level(similarity)
            }
            
        except Exception as e:
            logger.error(f"Error comparing images: {e}")
            return {"error": str(e)}
    
    async def _load_image(self, image_data: Union[str, bytes, np.ndarray]) -> Optional[np.ndarray]:
        """Load image from various input formats."""
        try:
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    # Base64 string
                    image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    return np.array(image)
                else:
                    # File path
                    image = cv2.imread(image_data)
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
            elif isinstance(image_data, bytes):
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
    
    async def _answer_with_blip(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Answer question using BLIP model."""
        try:
            inputs = self.processors['blip_vqa'](image, question, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.models['blip_vqa'].generate(**inputs, max_length=50)
            
            answer = self.processors['blip_vqa'].decode(outputs[0], skip_special_tokens=True)
            
            return {
                "answer": answer,
                "confidence": 0.8,  # BLIP doesn't provide confidence
                "model": "blip"
            }
            
        except Exception as e:
            return {"error": f"BLIP VQA failed: {e}"}
    
    async def _answer_with_vilt(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Answer question using ViLT model."""
        try:
            inputs = self.processors['vilt'](image, question, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.models['vilt'](**inputs)
                logits = outputs.logits
                predicted_answer = self.models['vilt'].config.id2label[logits.argmax(-1).item()]
            
            # Get confidence score
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probabilities.max().item()
            
            return {
                "answer": predicted_answer,
                "confidence": confidence,
                "model": "vilt"
            }
            
        except Exception as e:
            return {"error": f"ViLT VQA failed: {e}"}
    
    async def _answer_with_clip(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Answer question using CLIP model with predefined answers."""
        try:
            # For CLIP, we need to provide possible answers
            # This is a simplified approach - in practice, you'd want more sophisticated answer generation
            possible_answers = [
                "yes", "no", "person", "animal", "building", "car", "tree", "food",
                "indoor", "outdoor", "day", "night", "red", "blue", "green", "yellow",
                "one", "two", "few", "many", "large", "small", "none"
            ]
            
            # Process image
            image_input = self.processors['clip'](image).unsqueeze(0).to(self.device)
            
            # Process question + answers
            text_inputs = clip.tokenize([f"{question} {answer}" for answer in possible_answers]).to(self.device)
            
            with torch.no_grad():
                image_features = self.models['clip'].encode_image(image_input)
                text_features = self.models['clip'].encode_text(text_inputs)
                
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                best_answer_idx = similarities.argmax().item()
                confidence = similarities[0][best_answer_idx].item()
            
            return {
                "answer": possible_answers[best_answer_idx],
                "confidence": confidence,
                "model": "clip"
            }
            
        except Exception as e:
            return {"error": f"CLIP VQA failed: {e}"}
    
    async def _analyze_object_relationship(self, obj1: Dict, obj2: Dict) -> str:
        """Analyze spatial relationship between two objects."""
        try:
            box1 = obj1["box"]
            box2 = obj2["box"]
            
            # Calculate centers
            center1 = ((box1["xmin"] + box1["xmax"]) / 2, (box1["ymin"] + box1["ymax"]) / 2)
            center2 = ((box2["xmin"] + box2["xmax"]) / 2, (box2["ymin"] + box2["ymax"]) / 2)
            
            # Determine relationship
            if center1[0] < center2[0]:
                horizontal = "left of"
            elif center1[0] > center2[0]:
                horizontal = "right of"
            else:
                horizontal = "aligned with"
            
            if center1[1] < center2[1]:
                vertical = "above"
            elif center1[1] > center2[1]:
                vertical = "below"
            else:
                vertical = "level with"
            
            # Check for overlap
            overlap_x = max(0, min(box1["xmax"], box2["xmax"]) - max(box1["xmin"], box2["xmin"]))
            overlap_y = max(0, min(box1["ymax"], box2["ymax"]) - max(box1["ymin"], box2["ymin"]))
            
            if overlap_x > 0 and overlap_y > 0:
                return "overlapping with"
            
            # Combine relationships
            if horizontal == "aligned with" and vertical != "level with":
                return vertical
            elif vertical == "level with" and horizontal != "aligned with":
                return horizontal
            else:
                return f"{vertical} and {horizontal}"
                
        except Exception as e:
            logger.error(f"Error analyzing relationship: {e}")
            return "unknown relationship"
    
    async def _generate_structured_description(self, analysis: Dict) -> Dict[str, Any]:
        """Generate structured description from analysis results."""
        try:
            description = {
                "summary": analysis.get("caption", ""),
                "main_elements": [],
                "scene_context": {},
                "activities": []
            }
            
            # Extract main elements from objects
            if "objects" in analysis:
                description["main_elements"] = [obj["label"] for obj in analysis["objects"]]
            
            # Scene context
            description["scene_context"] = {
                "type": analysis.get("scene_type", "unknown"),
                "confidence": analysis.get("scene_confidence", 0)
            }
            
            # Extract activities from Q&A
            if "detailed_qa" in analysis:
                for qa in analysis["detailed_qa"]:
                    if "activity" in qa["question"].lower():
                        description["activities"].append(qa["answer"])
            
            return description
            
        except Exception as e:
            logger.error(f"Error generating structured description: {e}")
            return {"summary": "", "main_elements": [], "scene_context": {}, "activities": []}
    
    def _get_similarity_level(self, similarity: float) -> str:
        """Convert similarity score to human-readable level."""
        if similarity > 0.8:
            return "very similar"
        elif similarity > 0.6:
            return "similar"
        elif similarity > 0.4:
            return "somewhat similar"
        elif similarity > 0.2:
            return "different"
        else:
            return "very different"
    
    async def batch_vqa(self, image_data: Union[str, bytes, np.ndarray],
                       questions: List[str],
                       model: str = "blip") -> List[Dict[str, Any]]:
        """Process multiple questions for a single image."""
        tasks = [self.answer_question(image_data, question, model) for question in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            result if not isinstance(result, Exception) else {"error": str(result)}
            for result in results
        ]
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported VQA models."""
        return ["blip", "vilt", "clip"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "models_loaded": list(self.models.keys()),
            "processors_loaded": list(self.processors.keys()),
            "device": self.device,
            "vqa_deps_available": HAS_VQA_DEPS,
            "supported_models": self.get_supported_models()
        }