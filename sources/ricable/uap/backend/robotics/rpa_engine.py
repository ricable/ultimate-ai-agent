# backend/robotics/rpa_engine.py
# Agent 34: Advanced Robotics Integration - Robotic Process Automation Engine

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
from pathlib import Path

# Computer Vision Integration
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# GUI Automation
try:
    import pyautogui
    import psutil
    GUI_AUTOMATION_AVAILABLE = True
except ImportError:
    GUI_AUTOMATION_AVAILABLE = False

class RPATaskType(Enum):
    """Types of RPA tasks"""
    SCREEN_CAPTURE = "screen_capture"
    IMAGE_RECOGNITION = "image_recognition"
    CLICK_ACTION = "click_action"
    TYPE_TEXT = "type_text"
    SCROLL_ACTION = "scroll_action"
    WAIT_FOR_ELEMENT = "wait_for_element"
    CONDITIONAL_LOGIC = "conditional_logic"
    DATA_EXTRACTION = "data_extraction"
    FILE_OPERATION = "file_operation"
    CUSTOM_SCRIPT = "custom_script"

class RPATaskStatus(Enum):
    """Status of RPA tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"

@dataclass
class RPACoordinate:
    """Screen coordinate for RPA actions"""
    x: int
    y: int
    confidence: float = 1.0

@dataclass
class RPAElement:
    """UI element for RPA automation"""
    name: str
    element_type: str  # button, textbox, image, etc.
    coordinates: Optional[RPACoordinate]
    image_template: Optional[str]  # Base64 encoded template
    text_content: Optional[str]
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

@dataclass
class RPATask:
    """Individual RPA automation task"""
    task_id: str
    task_type: RPATaskType
    name: str
    description: str
    parameters: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0
    dependencies: List[str] = None
    status: RPATaskStatus = RPATaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class RPAWorkflow:
    """Complete RPA workflow with multiple tasks"""
    workflow_id: str
    name: str
    description: str
    tasks: List[RPATask]
    variables: Dict[str, Any] = None
    status: RPATaskStatus = RPATaskStatus.PENDING
    created_at: datetime = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class ScreenCapture:
    """Screen capture and analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def capture_screen(self, region: Optional[tuple] = None) -> np.ndarray:
        """Capture screen or region"""
        if not GUI_AUTOMATION_AVAILABLE:
            # Mock screen capture
            if region:
                return np.zeros((region[3], region[2], 3), dtype=np.uint8)
            return np.zeros((600, 800, 3), dtype=np.uint8)
        
        try:
            if region:
                # Capture specific region (x, y, width, height)
                screenshot = pyautogui.screenshot(region=region)
            else:
                # Capture full screen
                screenshot = pyautogui.screenshot()
            
            # Convert PIL to OpenCV format
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            return screenshot_cv
            
        except Exception as e:
            self.logger.error(f"Screen capture failed: {e}")
            # Return black image on failure
            return np.zeros((600, 800, 3), dtype=np.uint8)
    
    async def save_screenshot(self, image: np.ndarray, filename: str) -> str:
        """Save screenshot to file"""
        try:
            success = cv2.imwrite(filename, image)
            if success:
                return filename
            else:
                raise Exception("Failed to save image")
        except Exception as e:
            self.logger.error(f"Failed to save screenshot: {e}")
            return None

class ImageRecognition:
    """Image recognition for RPA automation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def find_template(self, screen: np.ndarray, template: np.ndarray, 
                           threshold: float = 0.8) -> List[RPACoordinate]:
        """Find template matches in screen image"""
        if not CV_AVAILABLE:
            # Mock template matching
            return [RPACoordinate(x=100, y=100, confidence=0.9)]
        
        try:
            # Template matching
            result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            
            matches = []
            for pt in zip(*locations[::-1]):
                confidence = float(result[pt[1], pt[0]])
                matches.append(RPACoordinate(x=pt[0], y=pt[1], confidence=confidence))
            
            # Remove duplicate matches
            return self._remove_duplicate_matches(matches, template.shape)
            
        except Exception as e:
            self.logger.error(f"Template matching failed: {e}")
            return []
    
    def _remove_duplicate_matches(self, matches: List[RPACoordinate], 
                                 template_shape: tuple, min_distance: int = 50) -> List[RPACoordinate]:
        """Remove duplicate/overlapping matches"""
        if not matches:
            return matches
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        filtered_matches = []
        for match in matches:
            is_duplicate = False
            for existing in filtered_matches:
                distance = ((match.x - existing.x) ** 2 + (match.y - existing.y) ** 2) ** 0.5
                if distance < min_distance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_matches.append(match)
        
        return filtered_matches
    
    async def find_text(self, screen: np.ndarray, target_text: str) -> List[RPACoordinate]:
        """Find text in screen using OCR"""
        try:
            # Import vision processor for OCR
            from ..vision.image_processor import vision_processor
            
            # Get OCR results
            ocr_results = await vision_processor.ocr_processor.extract_text(screen)
            
            matches = []
            for ocr_result in ocr_results:
                if target_text.lower() in ocr_result.text.lower():
                    matches.append(RPACoordinate(
                        x=ocr_result.bbox.x + ocr_result.bbox.width // 2,
                        y=ocr_result.bbox.y + ocr_result.bbox.height // 2,
                        confidence=ocr_result.confidence
                    ))
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Text recognition failed: {e}")
            return []

class MouseKeyboardController:
    """Mouse and keyboard automation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def click(self, coordinate: RPACoordinate, button: str = 'left', 
                   double_click: bool = False) -> bool:
        """Perform mouse click"""
        if not GUI_AUTOMATION_AVAILABLE:
            self.logger.info(f"Mock click at ({coordinate.x}, {coordinate.y})")
            return True
        
        try:
            if double_click:
                pyautogui.doubleClick(coordinate.x, coordinate.y, button=button)
            else:
                pyautogui.click(coordinate.x, coordinate.y, button=button)
            
            await asyncio.sleep(0.1)  # Small delay after click
            return True
            
        except Exception as e:
            self.logger.error(f"Click failed: {e}")
            return False
    
    async def drag(self, start: RPACoordinate, end: RPACoordinate, duration: float = 1.0) -> bool:
        """Perform drag operation"""
        if not GUI_AUTOMATION_AVAILABLE:
            self.logger.info(f"Mock drag from ({start.x}, {start.y}) to ({end.x}, {end.y})")
            return True
        
        try:
            pyautogui.drag(end.x - start.x, end.y - start.y, duration, 
                          startPoint=(start.x, start.y))
            return True
            
        except Exception as e:
            self.logger.error(f"Drag failed: {e}")
            return False
    
    async def type_text(self, text: str, interval: float = 0.1) -> bool:
        """Type text with specified interval between characters"""
        if not GUI_AUTOMATION_AVAILABLE:
            self.logger.info(f"Mock typing: {text}")
            return True
        
        try:
            pyautogui.typewrite(text, interval=interval)
            return True
            
        except Exception as e:
            self.logger.error(f"Text typing failed: {e}")
            return False
    
    async def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Perform scroll operation"""
        if not GUI_AUTOMATION_AVAILABLE:
            self.logger.info(f"Mock scroll {clicks} clicks")
            return True
        
        try:
            if x is not None and y is not None:
                pyautogui.scroll(clicks, x=x, y=y)
            else:
                pyautogui.scroll(clicks)
            return True
            
        except Exception as e:
            self.logger.error(f"Scroll failed: {e}")
            return False
    
    async def hotkey(self, *keys) -> bool:
        """Send hotkey combination"""
        if not GUI_AUTOMATION_AVAILABLE:
            self.logger.info(f"Mock hotkey: {'+'.join(keys)}")
            return True
        
        try:
            pyautogui.hotkey(*keys)
            return True
            
        except Exception as e:
            self.logger.error(f"Hotkey failed: {e}")
            return False

class RPATaskExecutor:
    """Execute individual RPA tasks"""
    
    def __init__(self):
        self.screen_capture = ScreenCapture()
        self.image_recognition = ImageRecognition()
        self.mouse_keyboard = MouseKeyboardController()
        self.logger = logging.getLogger(__name__)
    
    async def execute_task(self, task: RPATask, context: Dict[str, Any] = None) -> RPATask:
        """Execute a single RPA task"""
        if context is None:
            context = {}
        
        task.status = RPATaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        
        try:
            # Execute based on task type
            if task.task_type == RPATaskType.SCREEN_CAPTURE:
                result = await self._execute_screen_capture(task, context)
            elif task.task_type == RPATaskType.IMAGE_RECOGNITION:
                result = await self._execute_image_recognition(task, context)
            elif task.task_type == RPATaskType.CLICK_ACTION:
                result = await self._execute_click_action(task, context)
            elif task.task_type == RPATaskType.TYPE_TEXT:
                result = await self._execute_type_text(task, context)
            elif task.task_type == RPATaskType.SCROLL_ACTION:
                result = await self._execute_scroll_action(task, context)
            elif task.task_type == RPATaskType.WAIT_FOR_ELEMENT:
                result = await self._execute_wait_for_element(task, context)
            elif task.task_type == RPATaskType.DATA_EXTRACTION:
                result = await self._execute_data_extraction(task, context)
            else:
                raise Exception(f"Unsupported task type: {task.task_type}")
            
            task.result = result
            task.status = RPATaskStatus.COMPLETED
            
        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}")
            task.error_message = str(e)
            
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = RPATaskStatus.RETRY
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
            else:
                task.status = RPATaskStatus.FAILED
        
        task.completed_at = datetime.utcnow()
        return task
    
    async def _execute_screen_capture(self, task: RPATask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute screen capture task"""
        region = task.parameters.get('region')
        save_path = task.parameters.get('save_path')
        
        screenshot = await self.screen_capture.capture_screen(region)
        
        result = {
            'screenshot_shape': screenshot.shape,
            'capture_time': datetime.utcnow().isoformat()
        }
        
        if save_path:
            saved_path = await self.screen_capture.save_screenshot(screenshot, save_path)
            result['saved_path'] = saved_path
        
        # Store screenshot in context for other tasks
        context['last_screenshot'] = screenshot
        
        return result
    
    async def _execute_image_recognition(self, task: RPATask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image recognition task"""
        template_path = task.parameters.get('template_path')
        template_data = task.parameters.get('template_data')  # Base64 encoded
        threshold = task.parameters.get('threshold', 0.8)
        
        # Get current screen
        screenshot = context.get('last_screenshot')
        if screenshot is None:
            screenshot = await self.screen_capture.capture_screen()
        
        # Load template
        if template_path:
            template = cv2.imread(template_path)
        elif template_data:
            import base64
            template_bytes = base64.b64decode(template_data)
            template_array = np.frombuffer(template_bytes, dtype=np.uint8)
            template = cv2.imdecode(template_array, cv2.IMREAD_COLOR)
        else:
            raise Exception("No template provided")
        
        # Find matches
        matches = await self.image_recognition.find_template(screenshot, template, threshold)
        
        return {
            'matches_found': len(matches),
            'matches': [asdict(match) for match in matches]
        }
    
    async def _execute_click_action(self, task: RPATask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute click action task"""
        x = task.parameters.get('x')
        y = task.parameters.get('y')
        button = task.parameters.get('button', 'left')
        double_click = task.parameters.get('double_click', False)
        
        if x is None or y is None:
            raise Exception("Click coordinates not provided")
        
        coordinate = RPACoordinate(x=x, y=y)
        success = await self.mouse_keyboard.click(coordinate, button, double_click)
        
        return {
            'click_successful': success,
            'coordinates': {'x': x, 'y': y},
            'button': button
        }
    
    async def _execute_type_text(self, task: RPATask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute type text task"""
        text = task.parameters.get('text')
        interval = task.parameters.get('interval', 0.1)
        
        if not text:
            raise Exception("No text provided for typing")
        
        success = await self.mouse_keyboard.type_text(text, interval)
        
        return {
            'typing_successful': success,
            'text_length': len(text)
        }
    
    async def _execute_scroll_action(self, task: RPATask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scroll action task"""
        clicks = task.parameters.get('clicks', 3)
        x = task.parameters.get('x')
        y = task.parameters.get('y')
        
        success = await self.mouse_keyboard.scroll(clicks, x, y)
        
        return {
            'scroll_successful': success,
            'clicks': clicks
        }
    
    async def _execute_wait_for_element(self, task: RPATask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute wait for element task"""
        template_path = task.parameters.get('template_path')
        template_data = task.parameters.get('template_data')
        text_to_find = task.parameters.get('text_to_find')
        timeout = task.parameters.get('timeout', 30.0)
        check_interval = task.parameters.get('check_interval', 1.0)
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            screenshot = await self.screen_capture.capture_screen()
            
            # Check for template or text
            if template_path or template_data:
                # Load template
                if template_path:
                    template = cv2.imread(template_path)
                else:
                    import base64
                    template_bytes = base64.b64decode(template_data)
                    template_array = np.frombuffer(template_bytes, dtype=np.uint8)
                    template = cv2.imdecode(template_array, cv2.IMREAD_COLOR)
                
                matches = await self.image_recognition.find_template(screenshot, template)
                if matches:
                    return {
                        'element_found': True,
                        'wait_time': time.time() - start_time,
                        'matches': [asdict(match) for match in matches]
                    }
            
            elif text_to_find:
                matches = await self.image_recognition.find_text(screenshot, text_to_find)
                if matches:
                    return {
                        'element_found': True,
                        'wait_time': time.time() - start_time,
                        'text_matches': [asdict(match) for match in matches]
                    }
            
            await asyncio.sleep(check_interval)
        
        # Timeout reached
        return {
            'element_found': False,
            'wait_time': timeout,
            'timeout_reached': True
        }
    
    async def _execute_data_extraction(self, task: RPATask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data extraction task"""
        region = task.parameters.get('region')
        extract_text = task.parameters.get('extract_text', True)
        extract_images = task.parameters.get('extract_images', False)
        
        # Capture screen or region
        screenshot = await self.screen_capture.capture_screen(region)
        
        result = {
            'extraction_time': datetime.utcnow().isoformat()
        }
        
        if extract_text:
            # Import vision processor for OCR
            from ..vision.image_processor import vision_processor
            ocr_results = await vision_processor.ocr_processor.extract_text(screenshot)
            
            result['extracted_text'] = [
                {
                    'text': ocr.text,
                    'confidence': ocr.confidence,
                    'bbox': asdict(ocr.bbox)
                }
                for ocr in ocr_results
            ]
        
        if extract_images:
            # Save extracted region as image
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, screenshot)
                result['extracted_image_path'] = tmp_file.name
        
        return result

class RPAWorkflowEngine:
    """Execute complete RPA workflows"""
    
    def __init__(self):
        self.task_executor = RPATaskExecutor()
        self.logger = logging.getLogger(__name__)
        self.active_workflows: Dict[str, RPAWorkflow] = {}
    
    async def execute_workflow(self, workflow: RPAWorkflow) -> RPAWorkflow:
        """Execute complete RPA workflow"""
        self.active_workflows[workflow.workflow_id] = workflow
        workflow.status = RPATaskStatus.RUNNING
        
        context = workflow.variables.copy()
        
        try:
            # Execute tasks in dependency order
            completed_tasks = set()
            
            while len(completed_tasks) < len(workflow.tasks):
                progress_made = False
                
                for task in workflow.tasks:
                    if (task.task_id not in completed_tasks and 
                        all(dep_id in completed_tasks for dep_id in task.dependencies)):
                        
                        # Execute task
                        executed_task = await self.task_executor.execute_task(task, context)
                        
                        if executed_task.status == RPATaskStatus.COMPLETED:
                            completed_tasks.add(task.task_id)
                            progress_made = True
                            
                            # Update context with task results
                            if executed_task.result:
                                context[f"task_{task.task_id}_result"] = executed_task.result
                        
                        elif executed_task.status == RPATaskStatus.FAILED:
                            self.logger.error(f"Task {task.task_id} failed, stopping workflow")
                            workflow.status = RPATaskStatus.FAILED
                            return workflow
                
                if not progress_made:
                    self.logger.error("Workflow stuck, no progress made")
                    workflow.status = RPATaskStatus.FAILED
                    return workflow
                
                # Small delay between task executions
                await asyncio.sleep(0.1)
            
            workflow.status = RPATaskStatus.COMPLETED
            
        except Exception as e:
            self.logger.error(f"Workflow {workflow.workflow_id} failed: {e}")
            workflow.status = RPATaskStatus.FAILED
        
        finally:
            if workflow.workflow_id in self.active_workflows:
                del self.active_workflows[workflow.workflow_id]
        
        return workflow
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of running workflow"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            'workflow_id': workflow.workflow_id,
            'status': workflow.status.value,
            'total_tasks': len(workflow.tasks),
            'completed_tasks': len([t for t in workflow.tasks if t.status == RPATaskStatus.COMPLETED]),
            'failed_tasks': len([t for t in workflow.tasks if t.status == RPATaskStatus.FAILED]),
            'current_task': next((t.name for t in workflow.tasks if t.status == RPATaskStatus.RUNNING), None)
        }
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel running workflow"""
        workflow = self.active_workflows.get(workflow_id)
        if workflow:
            workflow.status = RPATaskStatus.CANCELLED
            return True
        return False

# Global RPA engine instance
rpa_engine = RPAWorkflowEngine()
