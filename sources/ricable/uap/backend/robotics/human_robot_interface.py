# backend/robotics/human_robot_interface.py
# Agent 34: Advanced Robotics Integration - Human-Robot Interaction System

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import threading

# NLP libraries
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Import robotics components
from .navigation_planner import NavigationGoal, Waypoint, NavigationMode, navigation_controller
from .robot_coordinator import Task, TaskType, TaskPriority, robot_coordinator
from .rpa_engine import RPATask, RPATaskType, rpa_engine
from .vision_processor import RobotVisionTask, robotics_vision_processor

class InteractionMode(Enum):
    """Types of human-robot interaction"""
    VOICE_COMMAND = "voice_command"
    TEXT_COMMAND = "text_command"
    GESTURE_RECOGNITION = "gesture_recognition"
    COLLABORATIVE_TASK = "collaborative_task"
    EMERGENCY_OVERRIDE = "emergency_override"

class IntentType(Enum):
    """Types of user intents"""
    NAVIGATION = "navigation"
    TASK_EXECUTION = "task_execution"
    QUESTION = "question"
    STATUS_INQUIRY = "status_inquiry"
    SAFETY_COMMAND = "safety_command"
    SYSTEM_CONTROL = "system_control"
    UNKNOWN = "unknown"

class ResponseType(Enum):
    """Types of robot responses"""
    ACKNOWLEDGMENT = "acknowledgment"
    STATUS_REPORT = "status_report"
    QUESTION_CLARIFICATION = "question_clarification"
    ERROR_REPORT = "error_report"
    COMPLETION_REPORT = "completion_report"

@dataclass
class HumanCommand:
    """Human command/input to robot"""
    command_id: str
    interaction_mode: InteractionMode
    raw_input: str
    processed_text: str
    intent: IntentType
    entities: Dict[str, Any]
    confidence: float
    timestamp: datetime
    user_id: Optional[str] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

@dataclass
class RobotResponse:
    """Robot response to human"""
    response_id: str
    command_id: str
    response_type: ResponseType
    message: str
    action_taken: Optional[str] = None
    additional_data: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.additional_data is None:
            self.additional_data = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class NaturalLanguageProcessor:
    """Natural language processing for robot commands"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nlp_pipeline = None
        self.intent_classifier = None
        
        # Intent patterns
        self.intent_patterns = {
            IntentType.NAVIGATION: [
                r"go to|move to|navigate to|drive to",
                r"position (\d+\.?\d*),?\s*(\d+\.?\d*)",
                r"coordinates|location|position"
            ],
            IntentType.TASK_EXECUTION: [
                r"pick up|grab|grasp|take",
                r"deliver|bring|carry",
                r"clean|sweep|vacuum",
                r"inspect|check|examine",
                r"perform|execute|do"
            ],
            IntentType.QUESTION: [
                r"what is|what are|how is|how are",
                r"where is|where are",
                r"when will|when did",
                r"why|how"
            ],
            IntentType.STATUS_INQUIRY: [
                r"status|state|condition",
                r"how are you|are you okay",
                r"what's happening|what's going on"
            ],
            IntentType.SAFETY_COMMAND: [
                r"stop|halt|emergency",
                r"slow down|be careful",
                r"safety|danger|hazard"
            ],
            IntentType.SYSTEM_CONTROL: [
                r"start|begin|initialize",
                r"stop|shutdown|turn off",
                r"restart|reset|reboot"
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'coordinates': r"(\d+\.?\d*),?\s*(\d+\.?\d*)",
            'object': r"(pick up|grab|take)\s+(?:the\s+)?(\w+)",
            'location': r"(room|area|zone|position)\s+(\w+)",
            'direction': r"(left|right|forward|backward|up|down)",
            'distance': r"(\d+\.?\d*)\s*(meter|meters|m|cm|centimeter|centimeters)",
            'speed': r"(slow|fast|normal|quick)\s*(speed|ly)?",
        }
        
        # Initialize NLP components
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP components"""
        if NLP_AVAILABLE:
            try:
                # Initialize sentiment analysis pipeline
                self.nlp_pipeline = pipeline("sentiment-analysis")
                self.logger.info("Initialized NLP pipeline")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NLP pipeline: {e}")
                self.nlp_pipeline = None
        else:
            self.logger.warning("NLP libraries not available, using rule-based processing")
    
    async def process_command(self, raw_input: str, interaction_mode: InteractionMode = InteractionMode.TEXT_COMMAND,
                            user_id: Optional[str] = None) -> HumanCommand:
        """Process human command and extract intent/entities"""
        command_id = f"cmd_{int(datetime.utcnow().timestamp() * 1000)}"
        
        # Clean and normalize input
        processed_text = self._clean_text(raw_input)
        
        # Extract intent
        intent, confidence = self._extract_intent(processed_text)
        
        # Extract entities
        entities = self._extract_entities(processed_text, intent)
        
        # Create command object
        command = HumanCommand(
            command_id=command_id,
            interaction_mode=interaction_mode,
            raw_input=raw_input,
            processed_text=processed_text,
            intent=intent,
            entities=entities,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            user_id=user_id
        )
        
        self.logger.info(f"Processed command: {intent.value} (confidence: {confidence:.2f})")
        return command
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text input"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation at the end
        text = re.sub(r'[.!?]+$', '', text)
        
        return text
    
    def _extract_intent(self, text: str) -> Tuple[IntentType, float]:
        """Extract intent from text"""
        best_intent = IntentType.UNKNOWN
        best_score = 0.0
        
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, text):
                    score += 1.0
            
            # Normalize score
            score = score / len(patterns)
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        # If no patterns matched, try NLP-based classification
        if best_score == 0.0 and self.nlp_pipeline:
            try:
                # Simple keyword-based fallback
                navigation_keywords = ['go', 'move', 'navigate', 'drive', 'position']
                task_keywords = ['pick', 'deliver', 'clean', 'inspect', 'do']
                question_keywords = ['what', 'where', 'how', 'when', 'why']
                
                if any(keyword in text for keyword in navigation_keywords):
                    best_intent = IntentType.NAVIGATION
                    best_score = 0.7
                elif any(keyword in text for keyword in task_keywords):
                    best_intent = IntentType.TASK_EXECUTION
                    best_score = 0.7
                elif any(keyword in text for keyword in question_keywords):
                    best_intent = IntentType.QUESTION
                    best_score = 0.7
                
            except Exception as e:
                self.logger.error(f"NLP intent classification failed: {e}")
        
        return best_intent, best_score
    
    def _extract_entities(self, text: str, intent: IntentType) -> Dict[str, Any]:
        """Extract entities from text based on intent"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                if entity_type == 'coordinates':
                    if len(matches[0]) == 2:
                        entities['target_x'] = float(matches[0][0])
                        entities['target_y'] = float(matches[0][1])
                elif entity_type == 'object':
                    if len(matches[0]) == 2:
                        entities['action'] = matches[0][0]
                        entities['target_object'] = matches[0][1]
                elif entity_type == 'distance':
                    if len(matches[0]) == 2:
                        value = float(matches[0][0])
                        unit = matches[0][1]
                        # Convert to meters
                        if unit in ['cm', 'centimeter', 'centimeters']:
                            value = value / 100.0
                        entities['distance'] = value
                else:
                    entities[entity_type] = matches[0] if len(matches[0]) == 1 else matches[0]
        
        # Intent-specific entity extraction
        if intent == IntentType.NAVIGATION:
            # Look for destination names
            destinations = ['kitchen', 'bedroom', 'office', 'garage', 'living room']
            for dest in destinations:
                if dest in text:
                    entities['destination'] = dest
                    break
        
        elif intent == IntentType.TASK_EXECUTION:
            # Look for task types
            if 'pick up' in text or 'grab' in text:
                entities['task_type'] = 'pickup'
            elif 'deliver' in text or 'bring' in text:
                entities['task_type'] = 'delivery'
            elif 'clean' in text:
                entities['task_type'] = 'cleaning'
            elif 'inspect' in text or 'check' in text:
                entities['task_type'] = 'inspection'
        
        return entities

class RobotDialogueManager:
    """Manage dialogue and responses"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conversation_history: List[Tuple[HumanCommand, RobotResponse]] = []
        self.context = {}
        
        # Response templates
        self.response_templates = {
            ResponseType.ACKNOWLEDGMENT: [
                "Understood. I will {action}.",
                "Roger that. Executing {action}.",
                "Acknowledged. Proceeding with {action}.",
                "Got it. Starting {action}."
            ],
            ResponseType.STATUS_REPORT: [
                "Current status: {status}",
                "I am currently {status}",
                "Status update: {status}"
            ],
            ResponseType.QUESTION_CLARIFICATION: [
                "Could you please clarify {question}?",
                "I need more information about {question}.",
                "Can you be more specific about {question}?"
            ],
            ResponseType.ERROR_REPORT: [
                "I encountered an error: {error}",
                "Sorry, I cannot complete that task: {error}",
                "There was a problem: {error}"
            ],
            ResponseType.COMPLETION_REPORT: [
                "Task completed successfully: {task}",
                "I have finished {task}",
                "Mission accomplished: {task}"
            ]
        }
    
    async def generate_response(self, command: HumanCommand) -> RobotResponse:
        """Generate appropriate response to human command"""
        response_id = f"resp_{int(datetime.utcnow().timestamp() * 1000)}"
        
        try:
            if command.intent == IntentType.NAVIGATION:
                return await self._handle_navigation_command(command, response_id)
            
            elif command.intent == IntentType.TASK_EXECUTION:
                return await self._handle_task_command(command, response_id)
            
            elif command.intent == IntentType.QUESTION:
                return await self._handle_question(command, response_id)
            
            elif command.intent == IntentType.STATUS_INQUIRY:
                return await self._handle_status_inquiry(command, response_id)
            
            elif command.intent == IntentType.SAFETY_COMMAND:
                return await self._handle_safety_command(command, response_id)
            
            elif command.intent == IntentType.SYSTEM_CONTROL:
                return await self._handle_system_control(command, response_id)
            
            else:
                return self._create_clarification_response(command, response_id)
                
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            return self._create_error_response(command, response_id, str(e))
    
    async def _handle_navigation_command(self, command: HumanCommand, response_id: str) -> RobotResponse:
        """Handle navigation commands"""
        entities = command.entities
        
        if 'target_x' in entities and 'target_y' in entities:
            # Navigate to coordinates
            target = Waypoint(
                x=entities['target_x'],
                y=entities['target_y'],
                z=entities.get('target_z', 0.0)
            )
            
            goal = NavigationGoal(target=target, priority=1)
            success = await navigation_controller.set_navigation_goal(goal)
            
            if success:
                action = f"navigate to coordinates ({entities['target_x']}, {entities['target_y']})"
                message = self._format_response(ResponseType.ACKNOWLEDGMENT, action=action)
                return RobotResponse(
                    response_id=response_id,
                    command_id=command.command_id,
                    response_type=ResponseType.ACKNOWLEDGMENT,
                    message=message,
                    action_taken="navigation_goal_set"
                )
            else:
                return self._create_error_response(command, response_id, "Failed to set navigation goal")
        
        elif 'destination' in entities:
            # Navigate to named location
            # Map destination names to coordinates (mock)
            destination_map = {
                'kitchen': (2.0, 3.0),
                'bedroom': (-1.0, 5.0),
                'office': (4.0, -2.0),
                'garage': (-3.0, -1.0),
                'living room': (0.0, 0.0)
            }
            
            dest_name = entities['destination']
            if dest_name in destination_map:
                coords = destination_map[dest_name]
                target = Waypoint(x=coords[0], y=coords[1])
                goal = NavigationGoal(target=target, priority=1)
                success = await navigation_controller.set_navigation_goal(goal)
                
                if success:
                    action = f"navigate to the {dest_name}"
                    message = self._format_response(ResponseType.ACKNOWLEDGMENT, action=action)
                    return RobotResponse(
                        response_id=response_id,
                        command_id=command.command_id,
                        response_type=ResponseType.ACKNOWLEDGMENT,
                        message=message,
                        action_taken="navigation_goal_set"
                    )
            
            return self._create_error_response(command, response_id, f"Unknown destination: {dest_name}")
        
        else:
            return self._create_clarification_response(command, response_id, "Where would you like me to go?")
    
    async def _handle_task_command(self, command: HumanCommand, response_id: str) -> RobotResponse:
        """Handle task execution commands"""
        entities = command.entities
        
        if 'task_type' in entities:
            task_type = entities['task_type']
            
            # Create robot coordination task
            if task_type == 'pickup':
                target_object = entities.get('target_object', 'object')
                task = Task(
                    task_id=f"pickup_{int(datetime.utcnow().timestamp() * 1000)}",
                    task_type=TaskType.PICKUP,
                    priority=TaskPriority.NORMAL,
                    description=f"Pick up {target_object}",
                    required_capabilities=['manipulation', 'vision'],
                    parameters={'target_object': target_object}
                )
                
            elif task_type == 'delivery':
                task = Task(
                    task_id=f"delivery_{int(datetime.utcnow().timestamp() * 1000)}",
                    task_type=TaskType.DELIVERY,
                    priority=TaskPriority.NORMAL,
                    description="Delivery task",
                    required_capabilities=['navigation', 'manipulation']
                )
                
            elif task_type == 'cleaning':
                task = Task(
                    task_id=f"cleaning_{int(datetime.utcnow().timestamp() * 1000)}",
                    task_type=TaskType.CLEANING,
                    priority=TaskPriority.NORMAL,
                    description="Cleaning task",
                    required_capabilities=['navigation', 'cleaning']
                )
                
            elif task_type == 'inspection':
                task = Task(
                    task_id=f"inspection_{int(datetime.utcnow().timestamp() * 1000)}",
                    task_type=TaskType.INSPECTION,
                    priority=TaskPriority.NORMAL,
                    description="Inspection task",
                    required_capabilities=['vision', 'navigation']
                )
            else:
                return self._create_error_response(command, response_id, f"Unknown task type: {task_type}")
            
            # Add task to coordinator
            success = await robot_coordinator.add_task(task)
            
            if success:
                action = f"execute {task_type} task"
                message = self._format_response(ResponseType.ACKNOWLEDGMENT, action=action)
                return RobotResponse(
                    response_id=response_id,
                    command_id=command.command_id,
                    response_type=ResponseType.ACKNOWLEDGMENT,
                    message=message,
                    action_taken="task_scheduled",
                    additional_data={'task_id': task.task_id}
                )
            else:
                return self._create_error_response(command, response_id, "Failed to schedule task")
        
        else:
            return self._create_clarification_response(command, response_id, "What task would you like me to perform?")
    
    async def _handle_question(self, command: HumanCommand, response_id: str) -> RobotResponse:
        """Handle questions from human"""
        text = command.processed_text
        
        if 'where' in text:
            # Location question
            status = navigation_controller.get_navigation_status()
            current_position = "unknown"  # Would get from actual robot pose
            
            message = f"I am currently at position {current_position}. Navigation status: {status.get('status', 'unknown')}"
            
        elif 'what' in text and ('doing' in text or 'status' in text):
            # Status question
            coord_status = robot_coordinator.get_system_status()
            nav_status = navigation_controller.get_navigation_status()
            
            message = f"I am currently {nav_status.get('status', 'idle')}. "
            if coord_status.get('total_tasks', 0) > 0:
                message += f"I have {coord_status['pending_tasks']} pending tasks."
            else:
                message += "No tasks pending."
                
        elif 'how' in text:
            # How question - provide capability info
            message = "I can navigate, manipulate objects, process images, and respond to voice commands. What would you like me to help with?"
            
        else:
            message = "I'm not sure how to answer that question. Could you be more specific?"
        
        return RobotResponse(
            response_id=response_id,
            command_id=command.command_id,
            response_type=ResponseType.STATUS_REPORT,
            message=message
        )
    
    async def _handle_status_inquiry(self, command: HumanCommand, response_id: str) -> RobotResponse:
        """Handle status inquiries"""
        # Get system status
        coord_status = robot_coordinator.get_system_status()
        nav_status = navigation_controller.get_navigation_status()
        
        status_parts = [
            f"System status: {coord_status.get('system_running', 'unknown')}",
            f"Navigation: {nav_status.get('status', 'unknown')}",
            f"Pending tasks: {coord_status.get('pending_tasks', 0)}"
        ]
        
        message = self._format_response(ResponseType.STATUS_REPORT, status=", ".join(status_parts))
        
        return RobotResponse(
            response_id=response_id,
            command_id=command.command_id,
            response_type=ResponseType.STATUS_REPORT,
            message=message,
            additional_data={
                'coordination_status': coord_status,
                'navigation_status': nav_status
            }
        )
    
    async def _handle_safety_command(self, command: HumanCommand, response_id: str) -> RobotResponse:
        """Handle safety commands"""
        text = command.processed_text
        
        if 'stop' in text or 'halt' in text or 'emergency' in text:
            # Emergency stop
            await robot_coordinator.stop_coordination()
            navigation_controller.set_navigation_mode(NavigationMode.EMERGENCY_STOP)
            
            message = "Emergency stop activated. All operations halted."
            action_taken = "emergency_stop"
            
        elif 'slow' in text:
            # Slow down
            message = "Reducing speed for safety."
            action_taken = "speed_reduction"
            
        else:
            message = "Safety mode activated. All operations proceeding with caution."
            action_taken = "safety_mode"
        
        return RobotResponse(
            response_id=response_id,
            command_id=command.command_id,
            response_type=ResponseType.ACKNOWLEDGMENT,
            message=message,
            action_taken=action_taken
        )
    
    async def _handle_system_control(self, command: HumanCommand, response_id: str) -> RobotResponse:
        """Handle system control commands"""
        text = command.processed_text
        
        if 'start' in text or 'begin' in text or 'initialize' in text:
            # Start systems
            await robot_coordinator.start_coordination()
            navigation_controller.start_navigation()
            
            message = "Systems started and ready for operation."
            action_taken = "system_start"
            
        elif 'stop' in text or 'shutdown' in text:
            # Stop systems
            await robot_coordinator.stop_coordination()
            navigation_controller.stop_navigation()
            
            message = "Systems shutting down safely."
            action_taken = "system_stop"
            
        elif 'restart' in text or 'reset' in text:
            # Restart systems
            message = "Restarting systems. Please wait."
            action_taken = "system_restart"
            
        else:
            message = "System control command received."
            action_taken = "system_command"
        
        return RobotResponse(
            response_id=response_id,
            command_id=command.command_id,
            response_type=ResponseType.ACKNOWLEDGMENT,
            message=message,
            action_taken=action_taken
        )
    
    def _create_clarification_response(self, command: HumanCommand, response_id: str, 
                                     question: str = None) -> RobotResponse:
        """Create clarification response"""
        if question is None:
            question = "what you'd like me to do"
        
        message = self._format_response(ResponseType.QUESTION_CLARIFICATION, question=question)
        
        return RobotResponse(
            response_id=response_id,
            command_id=command.command_id,
            response_type=ResponseType.QUESTION_CLARIFICATION,
            message=message
        )
    
    def _create_error_response(self, command: HumanCommand, response_id: str, error: str) -> RobotResponse:
        """Create error response"""
        message = self._format_response(ResponseType.ERROR_REPORT, error=error)
        
        return RobotResponse(
            response_id=response_id,
            command_id=command.command_id,
            response_type=ResponseType.ERROR_REPORT,
            message=message
        )
    
    def _format_response(self, response_type: ResponseType, **kwargs) -> str:
        """Format response using templates"""
        templates = self.response_templates.get(response_type, ["Default response."])
        template = templates[0]  # Use first template for now
        
        try:
            return template.format(**kwargs)
        except KeyError:
            return template

class HumanRobotInterface:
    """Main human-robot interaction interface"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nlp_processor = NaturalLanguageProcessor()
        self.dialogue_manager = RobotDialogueManager()
        
        # Interaction state
        self.active_interactions: Dict[str, HumanCommand] = {}
        self.interaction_history: List[Tuple[HumanCommand, RobotResponse]] = []
        
        # Statistics
        self.stats = {
            'total_interactions': 0,
            'successful_commands': 0,
            'failed_commands': 0,
            'clarification_requests': 0
        }
    
    async def process_human_input(self, raw_input: str, interaction_mode: InteractionMode = InteractionMode.TEXT_COMMAND,
                                user_id: Optional[str] = None) -> Tuple[HumanCommand, RobotResponse]:
        """Process human input and generate robot response"""
        try:
            # Process command
            command = await self.nlp_processor.process_command(raw_input, interaction_mode, user_id)
            
            # Store active interaction
            self.active_interactions[command.command_id] = command
            
            # Generate response
            response = await self.dialogue_manager.generate_response(command)
            
            # Store interaction in history
            self.interaction_history.append((command, response))
            
            # Update statistics
            self.stats['total_interactions'] += 1
            if response.response_type == ResponseType.ACKNOWLEDGMENT:
                self.stats['successful_commands'] += 1
            elif response.response_type == ResponseType.ERROR_REPORT:
                self.stats['failed_commands'] += 1
            elif response.response_type == ResponseType.QUESTION_CLARIFICATION:
                self.stats['clarification_requests'] += 1
            
            # Remove from active interactions
            if command.command_id in self.active_interactions:
                del self.active_interactions[command.command_id]
            
            self.logger.info(f"Processed interaction: {command.intent.value} -> {response.response_type.value}")
            
            return command, response
            
        except Exception as e:
            self.logger.error(f"Failed to process human input: {e}")
            
            # Create error command and response
            error_command = HumanCommand(
                command_id=f"error_{int(datetime.utcnow().timestamp() * 1000)}",
                interaction_mode=interaction_mode,
                raw_input=raw_input,
                processed_text=raw_input,
                intent=IntentType.UNKNOWN,
                entities={},
                confidence=0.0,
                timestamp=datetime.utcnow(),
                user_id=user_id
            )
            
            error_response = RobotResponse(
                response_id=f"error_resp_{int(datetime.utcnow().timestamp() * 1000)}",
                command_id=error_command.command_id,
                response_type=ResponseType.ERROR_REPORT,
                message=f"I'm sorry, I couldn't process that command: {str(e)}"
            )
            
            return error_command, error_response
    
    def get_interaction_stats(self) -> Dict[str, Any]:
        """Get interaction statistics"""
        return {
            **self.stats,
            'active_interactions': len(self.active_interactions),
            'history_length': len(self.interaction_history),
            'success_rate': self.stats['successful_commands'] / max(self.stats['total_interactions'], 1)
        }
    
    def get_recent_interactions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent interactions"""
        recent = self.interaction_history[-count:]
        return [
            {
                'command': asdict(command),
                'response': asdict(response)
            }
            for command, response in recent
        ]

# Global human-robot interface instance
human_robot_interface = HumanRobotInterface()