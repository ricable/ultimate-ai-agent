# edge/inference/edge_runtime.py
"""
Edge Inference Runtime
Optimized runtime for running inference models on edge devices.
"""

import asyncio
import time
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

# Import edge components
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelFormat(Enum):
    """Supported model formats for edge inference"""
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    TFLITE = "tflite"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"

class InferenceDevice(Enum):
    """Inference execution devices"""
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"  # Neural Processing Unit
    EDGE_TPU = "edge_tpu"  # Google Edge TPU
    APPLE_ANE = "apple_ane"  # Apple Neural Engine

@dataclass
class ModelConfig:
    """Configuration for edge inference model"""
    model_id: str
    name: str
    format: ModelFormat
    model_path: str
    input_shape: List[int]
    output_shape: List[int]
    device: InferenceDevice = InferenceDevice.CPU
    batch_size: int = 1
    precision: str = "fp32"  # fp32, fp16, int8
    optimization_level: str = "basic"  # basic, advanced, aggressive
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class InferenceRequest:
    """Request for model inference"""
    request_id: str
    model_id: str
    input_data: Any
    preprocessing: Optional[Dict[str, Any]] = None
    postprocessing: Optional[Dict[str, Any]] = None
    timeout_ms: int = 5000
    priority: str = "normal"  # low, normal, high, critical
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class InferenceResult:
    """Result of model inference"""
    request_id: str
    model_id: str
    success: bool
    output_data: Any = None
    error_message: Optional[str] = None
    inference_time_ms: float = 0.0
    queue_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    device_used: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'model_id': self.model_id,
            'success': self.success,
            'output_data': self.output_data,
            'error_message': self.error_message,
            'inference_time_ms': self.inference_time_ms,
            'queue_time_ms': self.queue_time_ms,
            'preprocessing_time_ms': self.preprocessing_time_ms,
            'postprocessing_time_ms': self.postprocessing_time_ms,
            'device_used': self.device_used,
            'metadata': self.metadata
        }

class EdgeInferenceModel:
    """Wrapper for edge inference models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_id = config.model_id
        self.model = None
        self.session = None
        self.is_loaded = False
        self.load_time = 0.0
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.avg_inference_time = 0.0
        self.last_used = time.time()
    
    async def load_model(self) -> bool:
        """Load model into memory"""
        if self.is_loaded:
            return True
        
        start_time = time.time()
        
        try:
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
            
            if self.config.format == ModelFormat.ONNX and ONNX_AVAILABLE:
                await self._load_onnx_model()
            elif self.config.format == ModelFormat.PYTORCH and TORCH_AVAILABLE:
                await self._load_pytorch_model()
            elif self.config.format == ModelFormat.TENSORFLOW and TF_AVAILABLE:
                await self._load_tensorflow_model()
            else:
                raise ValueError(f"Unsupported model format: {self.config.format.value}")
            
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            logger.info(f"Loaded model {self.model_id} in {self.load_time:.3f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            return False
    
    async def _load_onnx_model(self) -> None:
        """Load ONNX model"""
        providers = ['CPUExecutionProvider']
        
        if self.config.device == InferenceDevice.GPU:
            providers.insert(0, 'CUDAExecutionProvider')
        
        # Create session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if self.config.optimization_level == "aggressive":
            sess_options.enable_profiling = False
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
        
        self.session = ort.InferenceSession(
            self.config.model_path,
            sess_options,
            providers=providers
        )
        
        logger.debug(f"ONNX model loaded with providers: {self.session.get_providers()}")
    
    async def _load_pytorch_model(self) -> None:
        """Load PyTorch model"""
        device = torch.device('cpu')
        if self.config.device == InferenceDevice.GPU and torch.cuda.is_available():
            device = torch.device('cuda')
        
        self.model = torch.load(self.config.model_path, map_location=device)
        self.model.eval()
        
        # Apply optimizations
        if self.config.optimization_level in ["advanced", "aggressive"]:
            if hasattr(torch, 'jit'):
                # TorchScript optimization
                sample_input = torch.randn(self.config.input_shape).to(device)
                self.model = torch.jit.trace(self.model, sample_input)
        
        logger.debug(f"PyTorch model loaded on device: {device}")
    
    async def _load_tensorflow_model(self) -> None:
        """Load TensorFlow model"""
        if self.config.format == ModelFormat.TFLITE:
            # TensorFlow Lite for edge devices
            self.interpreter = tf.lite.Interpreter(model_path=self.config.model_path)
            self.interpreter.allocate_tensors()
        else:
            # Regular TensorFlow model
            self.model = tf.saved_model.load(self.config.model_path)
        
        logger.debug("TensorFlow model loaded")
    
    async def run_inference(self, input_data: Any) -> Tuple[Any, float]:
        """Run inference on input data"""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_id} not loaded")
        
        start_time = time.perf_counter()
        
        try:
            if self.config.format == ModelFormat.ONNX and self.session:
                output = await self._run_onnx_inference(input_data)
            elif self.config.format == ModelFormat.PYTORCH and self.model:
                output = await self._run_pytorch_inference(input_data)
            elif self.config.format == ModelFormat.TENSORFLOW:
                output = await self._run_tensorflow_inference(input_data)
            else:
                raise ValueError(f"No inference method for format: {self.config.format.value}")
            
            inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            # Update performance metrics
            self.inference_count += 1
            self.total_inference_time += inference_time
            self.avg_inference_time = self.total_inference_time / self.inference_count
            self.last_used = time.time()
            
            return output, inference_time
            
        except Exception as e:
            logger.error(f"Inference error for model {self.model_id}: {e}")
            raise
    
    async def _run_onnx_inference(self, input_data: Any) -> Any:
        """Run ONNX inference"""
        input_name = self.session.get_inputs()[0].name
        
        # Convert input data to numpy array if needed
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data, dtype=np.float32)
        
        # Ensure correct shape
        if input_data.shape != tuple(self.config.input_shape):
            input_data = input_data.reshape(self.config.input_shape)
        
        outputs = self.session.run(None, {input_name: input_data})
        return outputs[0] if len(outputs) == 1 else outputs
    
    async def _run_pytorch_inference(self, input_data: Any) -> Any:
        """Run PyTorch inference"""
        device = next(self.model.parameters()).device
        
        # Convert input to tensor
        if not isinstance(input_data, torch.Tensor):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
        else:
            input_tensor = input_data
        
        # Ensure correct shape and device
        if input_tensor.shape != tuple(self.config.input_shape):
            input_tensor = input_tensor.reshape(self.config.input_shape)
        
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return output.cpu().numpy()
    
    async def _run_tensorflow_inference(self, input_data: Any) -> Any:
        """Run TensorFlow inference"""
        if hasattr(self, 'interpreter'):  # TensorFlow Lite
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            # Set input tensor
            if not isinstance(input_data, np.ndarray):
                input_data = np.array(input_data, dtype=np.float32)
            
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output = self.interpreter.get_tensor(output_details[0]['index'])
            return output
        else:
            # Regular TensorFlow
            if not isinstance(input_data, tf.Tensor):
                input_data = tf.constant(input_data, dtype=tf.float32)
            
            output = self.model(input_data)
            return output.numpy()
    
    async def unload_model(self) -> None:
        """Unload model from memory"""
        self.model = None
        self.session = None
        self.is_loaded = False
        
        logger.info(f"Unloaded model {self.model_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        return {
            'model_id': self.model_id,
            'format': self.config.format.value,
            'device': self.config.device.value,
            'is_loaded': self.is_loaded,
            'load_time': self.load_time,
            'inference_count': self.inference_count,
            'total_inference_time': self.total_inference_time,
            'avg_inference_time': self.avg_inference_time,
            'last_used': self.last_used,
            'input_shape': self.config.input_shape,
            'output_shape': self.config.output_shape
        }

class EdgeInferenceRuntime:
    """Edge inference runtime for managing multiple models"""
    
    def __init__(self, 
                 max_models: int = 10,
                 max_queue_size: int = 1000,
                 worker_threads: int = 4):
        
        self.max_models = max_models
        self.max_queue_size = max_queue_size
        self.worker_threads = worker_threads
        
        # Model management
        self.models: Dict[str, EdgeInferenceModel] = {}
        self.model_locks: Dict[str, threading.Lock] = {}
        
        # Request processing
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_tasks = []
        self.is_running = False
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'avg_queue_time_ms': 0.0,
            'avg_inference_time_ms': 0.0,
            'requests_per_second': 0.0,
            'models_loaded': 0,
            'memory_usage_mb': 0.0
        }
        
        logger.info(f"EdgeInferenceRuntime initialized: max_models={max_models}, "
                   f"queue_size={max_queue_size}, workers={worker_threads}")
    
    async def start(self) -> None:
        """Start the inference runtime"""
        if self.is_running:
            return
        
        logger.info("Starting EdgeInferenceRuntime")
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.worker_threads):
            task = asyncio.create_task(self._inference_worker(f"worker-{i}"))
            self.processing_tasks.append(task)
        
        # Start metrics collection
        metrics_task = asyncio.create_task(self._metrics_collector())
        self.processing_tasks.append(metrics_task)
        
        logger.info(f"Started {self.worker_threads} inference workers")
    
    async def load_model(self, config: ModelConfig) -> bool:
        """Load a model into the runtime"""
        if len(self.models) >= self.max_models:
            # Unload least recently used model
            await self._unload_lru_model()
        
        try:
            model = EdgeInferenceModel(config)
            success = await model.load_model()
            
            if success:
                self.models[config.model_id] = model
                self.model_locks[config.model_id] = threading.Lock()
                self.metrics['models_loaded'] += 1
                
                logger.info(f"Loaded model {config.model_id} into runtime")
                return True
            else:
                logger.error(f"Failed to load model {config.model_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {config.model_id}: {e}")
            return False
    
    async def run_inference(self, request: InferenceRequest) -> InferenceResult:
        """Run inference on a request"""
        start_time = time.perf_counter()
        
        try:
            # Add request to queue
            await self.request_queue.put((request, start_time))
            self.metrics['total_requests'] += 1
            
            # The actual processing happens in worker threads
            # This is a simplified implementation - in production,
            # you'd want to use a proper async request/response mechanism
            
            # For now, we'll process directly
            return await self._process_inference_request(request, start_time)
            
        except Exception as e:
            logger.error(f"Error running inference: {e}")
            self.metrics['failed_inferences'] += 1
            
            return InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                success=False,
                error_message=str(e)
            )
    
    async def _process_inference_request(self, request: InferenceRequest, queue_start_time: float) -> InferenceResult:
        """Process a single inference request"""
        processing_start_time = time.perf_counter()
        queue_time_ms = (processing_start_time - queue_start_time) * 1000
        
        try:
            # Check if model exists
            if request.model_id not in self.models:
                raise ValueError(f"Model {request.model_id} not found")
            
            model = self.models[request.model_id]
            
            # Preprocessing
            preprocessing_start = time.perf_counter()
            processed_input = await self._preprocess_input(request.input_data, request.preprocessing)
            preprocessing_time_ms = (time.perf_counter() - preprocessing_start) * 1000
            
            # Run inference
            inference_start = time.perf_counter()
            raw_output, inference_time_ms = await model.run_inference(processed_input)
            
            # Postprocessing
            postprocessing_start = time.perf_counter()
            final_output = await self._postprocess_output(raw_output, request.postprocessing)
            postprocessing_time_ms = (time.perf_counter() - postprocessing_start) * 1000
            
            # Update metrics
            self.metrics['successful_inferences'] += 1
            
            return InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                success=True,
                output_data=final_output,
                inference_time_ms=inference_time_ms,
                queue_time_ms=queue_time_ms,
                preprocessing_time_ms=preprocessing_time_ms,
                postprocessing_time_ms=postprocessing_time_ms,
                device_used=model.config.device.value,
                metadata={
                    'model_format': model.config.format.value,
                    'input_shape': model.config.input_shape,
                    'output_shape': model.config.output_shape
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing inference request {request.request_id}: {e}")
            self.metrics['failed_inferences'] += 1
            
            return InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                success=False,
                error_message=str(e),
                queue_time_ms=queue_time_ms
            )
    
    async def _preprocess_input(self, input_data: Any, preprocessing_config: Optional[Dict[str, Any]]) -> Any:
        """Preprocess input data"""
        if not preprocessing_config:
            return input_data
        
        processed_data = input_data
        
        # Apply preprocessing steps
        if 'normalize' in preprocessing_config:
            if NUMPY_AVAILABLE:
                processed_data = np.array(processed_data, dtype=np.float32)
                processed_data = processed_data / 255.0  # Common normalization
        
        if 'resize' in preprocessing_config:
            # Image resizing would go here
            pass
        
        if 'convert_type' in preprocessing_config:
            target_type = preprocessing_config['convert_type']
            if target_type == 'float32' and NUMPY_AVAILABLE:
                processed_data = np.array(processed_data, dtype=np.float32)
        
        return processed_data
    
    async def _postprocess_output(self, output_data: Any, postprocessing_config: Optional[Dict[str, Any]]) -> Any:
        """Postprocess output data"""
        if not postprocessing_config:
            return output_data
        
        processed_output = output_data
        
        # Apply postprocessing steps
        if 'softmax' in postprocessing_config and NUMPY_AVAILABLE:
            if isinstance(processed_output, np.ndarray):
                exp_scores = np.exp(processed_output - np.max(processed_output))
                processed_output = exp_scores / np.sum(exp_scores)
        
        if 'argmax' in postprocessing_config and NUMPY_AVAILABLE:
            if isinstance(processed_output, np.ndarray):
                processed_output = np.argmax(processed_output)
        
        if 'threshold' in postprocessing_config:
            threshold = postprocessing_config['threshold']
            if NUMPY_AVAILABLE and isinstance(processed_output, np.ndarray):
                processed_output = (processed_output > threshold).astype(int)
        
        return processed_output
    
    async def _inference_worker(self, worker_id: str) -> None:
        """Background worker for processing inference requests"""
        logger.info(f"Starting inference worker: {worker_id}")
        
        while self.is_running:
            try:
                # Get request from queue
                request, queue_start_time = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=1.0
                )
                
                # Process request
                result = await self._process_inference_request(request, queue_start_time)
                
                # In a full implementation, you'd send the result back to the client
                # via a response queue or callback mechanism
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)
    
    async def _unload_lru_model(self) -> None:
        """Unload least recently used model"""
        if not self.models:
            return
        
        # Find least recently used model
        lru_model_id = min(self.models.keys(), key=lambda mid: self.models[mid].last_used)
        
        # Unload model
        await self.models[lru_model_id].unload_model()
        del self.models[lru_model_id]
        del self.model_locks[lru_model_id]
        
        self.metrics['models_loaded'] -= 1
        logger.info(f"Unloaded LRU model: {lru_model_id}")
    
    async def _metrics_collector(self) -> None:
        """Background metrics collection"""
        last_request_count = 0
        last_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                current_request_count = self.metrics['total_requests']
                
                # Calculate requests per second
                time_diff = current_time - last_time
                if time_diff >= 1.0:
                    request_diff = current_request_count - last_request_count
                    self.metrics['requests_per_second'] = request_diff / time_diff
                    
                    last_request_count = current_request_count
                    last_time = current_time
                
                # Update average metrics
                if self.metrics['successful_inferences'] > 0:
                    total_inference_time = sum(
                        model.total_inference_time for model in self.models.values()
                    )
                    self.metrics['avg_inference_time_ms'] = (
                        total_inference_time / self.metrics['successful_inferences']
                    )
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5.0)
    
    def list_models(self) -> List[str]:
        """List all loaded models"""
        return list(self.models.keys())
    
    def get_model_stats(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific model"""
        if model_id in self.models:
            return self.models[model_id].get_stats()
        return None
    
    def get_runtime_metrics(self) -> Dict[str, Any]:
        """Get runtime performance metrics"""
        return {
            **self.metrics,
            'queue_size': self.request_queue.qsize(),
            'models_loaded': len(self.models),
            'worker_threads': len(self.processing_tasks) - 1,  # Exclude metrics task
            'is_running': self.is_running,
            'model_stats': {
                model_id: model.get_stats()
                for model_id, model in self.models.items()
            }
        }
    
    async def stop(self) -> None:
        """Stop the inference runtime"""
        logger.info("Stopping EdgeInferenceRuntime")
        
        self.is_running = False
        
        # Stop all processing tasks
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Unload all models
        for model in self.models.values():
            await model.unload_model()
        
        self.models.clear()
        self.model_locks.clear()
        
        logger.info("EdgeInferenceRuntime stopped")

# Global edge inference runtime
edge_runtime = EdgeInferenceRuntime()

# Convenience functions
async def load_edge_model(model_id: str, 
                         model_path: str,
                         format: str = "onnx",
                         input_shape: List[int] = [1, 3, 224, 224],
                         output_shape: List[int] = [1, 1000],
                         device: str = "cpu") -> bool:
    """Load a model for edge inference"""
    config = ModelConfig(
        model_id=model_id,
        name=f"Model {model_id}",
        format=ModelFormat(format),
        model_path=model_path,
        input_shape=input_shape,
        output_shape=output_shape,
        device=InferenceDevice(device)
    )
    
    return await edge_runtime.load_model(config)

async def run_edge_inference(model_id: str, 
                           input_data: Any,
                           preprocessing: Optional[Dict[str, Any]] = None,
                           postprocessing: Optional[Dict[str, Any]] = None) -> InferenceResult:
    """Run inference on edge device"""
    request = InferenceRequest(
        request_id=str(uuid.uuid4()),
        model_id=model_id,
        input_data=input_data,
        preprocessing=preprocessing,
        postprocessing=postprocessing
    )
    
    return await edge_runtime.run_inference(request)

def get_edge_runtime_metrics() -> Dict[str, Any]:
    """Get edge runtime metrics"""
    return edge_runtime.get_runtime_metrics()

def list_edge_models() -> List[str]:
    """List loaded edge models"""
    return edge_runtime.list_models()
