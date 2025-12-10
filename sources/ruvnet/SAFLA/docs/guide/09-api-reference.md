# SAFLA API Reference

## Overview

The SAFLA (Self-Aware Feedback Loop Algorithm) API provides a comprehensive interface for implementing autonomous systems with recursive, self-aware feedback loops. This reference covers all classes, methods, and parameters available in the SAFLA framework.

## Core Modules

### safla.core.meta_cognitive_engine

The meta-cognitive engine is the central coordination layer of SAFLA, implementing self-awareness and adaptive behavior.

#### Classes

##### MetaCognitiveEngine

The main coordination class that orchestrates all meta-cognitive components.

```python
class MetaCognitiveEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
```

**Parameters:**
- `config` (Dict[str, Any], optional): Configuration dictionary for engine initialization

**Methods:**

###### initialize()
```python
async def initialize() -> bool
```
Initialize the meta-cognitive engine and all its components.

**Returns:**
- `bool`: True if initialization successful, False otherwise

###### process_feedback()
```python
async def process_feedback(self, feedback: Dict[str, Any]) -> AdaptationResult
```
Process feedback through the meta-cognitive loop.

**Parameters:**
- `feedback` (Dict[str, Any]): Feedback data to process

**Returns:**
- `AdaptationResult`: Result of the adaptation process

###### get_system_state()
```python
def get_system_state() -> SystemState
```
Get the current system state.

**Returns:**
- `SystemState`: Current state of the system

##### SystemState

Data structure representing the current state of the system.

```python
@dataclass
class SystemState:
    state_id: str
    timestamp: float
    performance_metrics: Dict[str, float]
    active_goals: List[str]
    current_strategy: Optional[str]
    confidence_level: float
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Attributes:**
- `state_id` (str): Unique identifier for the state
- `timestamp` (float): Unix timestamp when state was captured
- `performance_metrics` (Dict[str, float]): Current performance metrics
- `active_goals` (List[str]): List of currently active goal IDs
- `current_strategy` (Optional[str]): ID of the currently active strategy
- `confidence_level` (float): System confidence level (0.0 to 1.0)
- `metadata` (Dict[str, Any]): Additional state metadata

##### Goal

Represents a system goal with success criteria and priority.

```python
@dataclass
class Goal:
    goal_id: str
    description: str
    target_metrics: Dict[str, float]
    priority: int
    deadline: Optional[float]
    success_criteria: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Attributes:**
- `goal_id` (str): Unique identifier for the goal
- `description` (str): Human-readable description
- `target_metrics` (Dict[str, float]): Target performance metrics
- `priority` (int): Goal priority (higher values = higher priority)
- `deadline` (Optional[float]): Optional deadline as Unix timestamp
- `success_criteria` (Dict[str, Any]): Criteria for goal completion
- `metadata` (Dict[str, Any]): Additional goal metadata

##### Strategy

Represents an execution strategy with associated actions.

```python
@dataclass
class Strategy:
    strategy_id: str
    name: str
    description: str
    actions: List[Dict[str, Any]]
    expected_outcomes: Dict[str, float]
    confidence_score: float
    resource_requirements: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Attributes:**
- `strategy_id` (str): Unique identifier for the strategy
- `name` (str): Strategy name
- `description` (str): Detailed description
- `actions` (List[Dict[str, Any]]): List of actions to execute
- `expected_outcomes` (Dict[str, float]): Expected performance outcomes
- `confidence_score` (float): Confidence in strategy success (0.0 to 1.0)
- `resource_requirements` (Dict[str, float]): Required resources
- `metadata` (Dict[str, Any]): Additional strategy metadata

##### PerformanceMetrics

Container for system performance metrics.

```python
@dataclass
class PerformanceMetrics:
    timestamp: float
    accuracy: float
    efficiency: float
    stability: float
    capability_score: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)
```

**Attributes:**
- `timestamp` (float): When metrics were captured
- `accuracy` (float): System accuracy score (0.0 to 1.0)
- `efficiency` (float): System efficiency score (0.0 to 1.0)
- `stability` (float): System stability score (0.0 to 1.0)
- `capability_score` (float): Overall capability score (0.0 to 1.0)
- `custom_metrics` (Dict[str, float]): Additional custom metrics

##### AdaptationResult

Result of an adaptation process.

```python
@dataclass
class AdaptationResult:
    adaptation_id: str
    timestamp: float
    success: bool
    changes_made: List[Dict[str, Any]]
    performance_impact: Dict[str, float]
    confidence_level: float
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Attributes:**
- `adaptation_id` (str): Unique identifier for the adaptation
- `timestamp` (float): When adaptation was performed
- `success` (bool): Whether adaptation was successful
- `changes_made` (List[Dict[str, Any]]): List of changes made
- `performance_impact` (Dict[str, float]): Impact on performance metrics
- `confidence_level` (float): Confidence in adaptation success
- `metadata` (Dict[str, Any]): Additional adaptation metadata

### safla.core.delta_evaluation

The delta evaluation system provides formal quantification of system improvements.

#### Classes

##### DeltaEvaluator

Main class for calculating delta improvements using the SAFLA formula.

```python
class DeltaEvaluator:
    def __init__(self, weights: Optional[AdaptiveWeights] = None)
```

**Parameters:**
- `weights` (AdaptiveWeights, optional): Custom weights for delta calculation

**Methods:**

###### calculate_delta()
```python
def calculate_delta(
    self, 
    current_metrics: PerformanceMetrics, 
    previous_metrics: PerformanceMetrics
) -> DeltaResult
```
Calculate the total delta improvement using the SAFLA formula:
Δ_total = α₁ × Δ_performance + α₂ × Δ_efficiency + α₃ × Δ_stability + α₄ × Δ_capability

**Parameters:**
- `current_metrics` (PerformanceMetrics): Current performance metrics
- `previous_metrics` (PerformanceMetrics): Previous performance metrics

**Returns:**
- `DeltaResult`: Comprehensive delta calculation result

###### update_weights()
```python
def update_weights(self, new_weights: AdaptiveWeights) -> None
```
Update the adaptive weights used in delta calculation.

**Parameters:**
- `new_weights` (AdaptiveWeights): New weights to apply

##### DeltaResult

Result of a delta evaluation calculation.

```python
@dataclass
class DeltaResult:
    total_delta: float
    component_deltas: DeltaMetrics
    weights_used: AdaptiveWeights
    calculation_timestamp: float
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Attributes:**
- `total_delta` (float): Total calculated delta value
- `component_deltas` (DeltaMetrics): Individual component deltas
- `weights_used` (AdaptiveWeights): Weights used in calculation
- `calculation_timestamp` (float): When calculation was performed
- `confidence_score` (float): Confidence in the calculation
- `metadata` (Dict[str, Any]): Additional calculation metadata

##### DeltaMetrics

Individual delta components.

```python
@dataclass
class DeltaMetrics:
    performance_delta: float
    efficiency_delta: float
    stability_delta: float
    capability_delta: float
```

**Attributes:**
- `performance_delta` (float): Change in performance
- `efficiency_delta` (float): Change in efficiency
- `stability_delta` (float): Change in stability
- `capability_delta` (float): Change in capability

##### AdaptiveWeights

Weights used in delta calculation that can adapt over time.

```python
@dataclass
class AdaptiveWeights:
    alpha_1: float = 0.25  # Performance weight
    alpha_2: float = 0.25  # Efficiency weight
    alpha_3: float = 0.25  # Stability weight
    alpha_4: float = 0.25  # Capability weight
    adaptation_rate: float = 0.1
    last_updated: float = field(default_factory=time.time)
```

**Attributes:**
- `alpha_1` (float): Weight for performance component
- `alpha_2` (float): Weight for efficiency component
- `alpha_3` (float): Weight for stability component
- `alpha_4` (float): Weight for capability component
- `adaptation_rate` (float): Rate at which weights adapt
- `last_updated` (float): Timestamp of last weight update

### safla.core.hybrid_memory

The hybrid memory architecture provides multi-type memory management with consolidation.

#### Classes

##### HybridMemoryArchitecture

Main memory management system integrating multiple memory types.

```python
class HybridMemoryArchitecture:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
```

**Parameters:**
- `config` (Dict[str, Any], optional): Memory configuration parameters

**Methods:**

###### store_memory()
```python
async def store_memory(
    self, 
    content: Any, 
    memory_type: str, 
    metadata: Optional[Dict[str, Any]] = None
) -> str
```
Store content in the appropriate memory system.

**Parameters:**
- `content` (Any): Content to store
- `memory_type` (str): Type of memory ("vector", "episodic", "semantic", "working")
- `metadata` (Dict[str, Any], optional): Additional metadata

**Returns:**
- `str`: Unique identifier for the stored memory

###### retrieve_memory()
```python
async def retrieve_memory(
    self, 
    memory_id: str, 
    memory_type: str
) -> Optional[MemoryItem]
```
Retrieve a specific memory item.

**Parameters:**
- `memory_id` (str): Unique identifier of the memory
- `memory_type` (str): Type of memory to search

**Returns:**
- `Optional[MemoryItem]`: Retrieved memory item or None if not found

###### search_similar()
```python
async def search_similar(
    self, 
    query_vector: List[float], 
    memory_type: str = "vector",
    max_results: int = 10,
    similarity_threshold: float = 0.8
) -> List[MemoryItem]
```
Search for similar memories using vector similarity.

**Parameters:**
- `query_vector` (List[float]): Query vector for similarity search
- `memory_type` (str): Type of memory to search
- `max_results` (int): Maximum number of results
- `similarity_threshold` (float): Minimum similarity threshold

**Returns:**
- `List[MemoryItem]`: List of similar memory items

##### MemoryItem

Base class for all memory items.

```python
@dataclass
class MemoryItem:
    memory_id: str
    content: Any
    embedding: Optional[List[float]]
    timestamp: float
    access_count: int = 0
    importance_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Attributes:**
- `memory_id` (str): Unique identifier
- `content` (Any): Stored content
- `embedding` (Optional[List[float]]): Vector embedding
- `timestamp` (float): Creation timestamp
- `access_count` (int): Number of times accessed
- `importance_score` (float): Importance score (0.0 to 1.0)
- `metadata` (Dict[str, Any]): Additional metadata

##### VectorMemoryManager

Manages vector-based memory storage and retrieval.

```python
class VectorMemoryManager:
    def __init__(self, embedding_dimension: int = 512)
```

**Parameters:**
- `embedding_dimension` (int): Dimension of embedding vectors

**Methods:**

###### store_vector()
```python
async def store_vector(
    self, 
    content: Any, 
    embedding: List[float], 
    metadata: Optional[Dict[str, Any]] = None
) -> str
```
Store content with its vector embedding.

**Parameters:**
- `content` (Any): Content to store
- `embedding` (List[float]): Vector embedding
- `metadata` (Dict[str, Any], optional): Additional metadata

**Returns:**
- `str`: Unique identifier for stored vector

###### similarity_search()
```python
async def similarity_search(
    self, 
    query_vector: List[float], 
    max_results: int = 10,
    similarity_threshold: float = 0.8
) -> List[Tuple[MemoryItem, float]]
```
Perform similarity search using cosine similarity.

**Parameters:**
- `query_vector` (List[float]): Query vector
- `max_results` (int): Maximum results to return
- `similarity_threshold` (float): Minimum similarity threshold

**Returns:**
- `List[Tuple[MemoryItem, float]]`: List of (memory_item, similarity_score) tuples

### safla.core.safety_validation

The safety validation framework provides comprehensive safety mechanisms.

#### Classes

##### SafetyValidationFramework

Main safety framework coordinating all safety components.

```python
class SafetyValidationFramework:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
```

**Parameters:**
- `config` (Dict[str, Any], optional): Safety configuration parameters

**Methods:**

###### validate_action()
```python
async def validate_action(self, action: Dict[str, Any]) -> ValidationResult
```
Validate an action against safety constraints.

**Parameters:**
- `action` (Dict[str, Any]): Action to validate

**Returns:**
- `ValidationResult`: Validation result with safety assessment

###### assess_risk()
```python
async def assess_risk(
    self, 
    context: Dict[str, Any], 
    proposed_action: Dict[str, Any]
) -> RiskAssessment
```
Assess risk level of a proposed action.

**Parameters:**
- `context` (Dict[str, Any]): Current system context
- `proposed_action` (Dict[str, Any]): Action to assess

**Returns:**
- `RiskAssessment`: Comprehensive risk assessment

###### create_checkpoint()
```python
async def create_checkpoint(self, description: str) -> str
```
Create a system checkpoint for rollback capability.

**Parameters:**
- `description` (str): Description of the checkpoint

**Returns:**
- `str`: Unique checkpoint identifier

###### rollback_to_checkpoint()
```python
async def rollback_to_checkpoint(self, checkpoint_id: str) -> bool
```
Rollback system to a previous checkpoint.

**Parameters:**
- `checkpoint_id` (str): Identifier of checkpoint to rollback to

**Returns:**
- `bool`: True if rollback successful, False otherwise

##### ValidationResult

Result of a safety validation check.

```python
@dataclass
class ValidationResult:
    is_valid: bool
    risk_level: str
    violations: List[ConstraintViolation]
    recommendations: List[str]
    confidence_score: float
    validation_timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Attributes:**
- `is_valid` (bool): Whether action passes validation
- `risk_level` (str): Risk level ("low", "medium", "high", "critical")
- `violations` (List[ConstraintViolation]): List of constraint violations
- `recommendations` (List[str]): Safety recommendations
- `confidence_score` (float): Confidence in validation result
- `validation_timestamp` (float): When validation was performed
- `metadata` (Dict[str, Any]): Additional validation metadata

##### RiskAssessment

Comprehensive risk assessment result.

```python
@dataclass
class RiskAssessment:
    overall_risk_score: float
    risk_categories: Dict[str, float]
    mitigation_strategies: List[str]
    assessment_confidence: float
    assessment_timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Attributes:**
- `overall_risk_score` (float): Overall risk score (0.0 to 1.0)
- `risk_categories` (Dict[str, float]): Risk scores by category
- `mitigation_strategies` (List[str]): Recommended mitigation strategies
- `assessment_confidence` (float): Confidence in assessment
- `assessment_timestamp` (float): When assessment was performed
- `metadata` (Dict[str, Any]): Additional assessment metadata

### safla.core.mcp_orchestration

The MCP orchestration infrastructure manages Model Context Protocol servers and agent coordination.

#### Classes

##### MCPOrchestrator

Main orchestrator coordinating all MCP infrastructure components.

```python
class MCPOrchestrator:
    def __init__(self)
```

**Methods:**

###### start()
```python
async def start() -> None
```
Start the MCP orchestrator and all its components.

###### shutdown()
```python
async def shutdown() -> None
```
Shutdown the MCP orchestrator gracefully.

###### process_request()
```python
async def process_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]
```
Process a request through the orchestration system.

**Parameters:**
- `request` (Dict[str, Any]): Request to process

**Returns:**
- `Optional[Dict[str, Any]]`: Response from processing or None if failed

###### get_system_status()
```python
def get_system_status() -> Dict[str, Any]
```
Get comprehensive system status information.

**Returns:**
- `Dict[str, Any]`: System status including servers, agents, and resources

##### MCPServer

Data structure representing an MCP server.

```python
@dataclass
class MCPServer:
    server_id: str
    name: str
    endpoint: str
    capabilities: List[str]
    status: MCPServerStatus
    health_score: float = 1.0
    load_factor: float = 0.0
    last_health_check: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Attributes:**
- `server_id` (str): Unique server identifier
- `name` (str): Human-readable server name
- `endpoint` (str): Server endpoint URL
- `capabilities` (List[str]): List of server capabilities
- `status` (MCPServerStatus): Current server status
- `health_score` (float): Health score (0.0 to 1.0)
- `load_factor` (float): Current load factor (0.0 to 1.0)
- `last_health_check` (Optional[float]): Last health check timestamp
- `metadata` (Dict[str, Any]): Additional server metadata

##### Agent

Data structure representing an agent in the system.

```python
@dataclass
class Agent:
    agent_id: str
    name: str
    agent_type: str
    capabilities: List[str]
    status: AgentStatus
    server_id: str
    priority: int = 1
    resource_allocation: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Attributes:**
- `agent_id` (str): Unique agent identifier
- `name` (str): Human-readable agent name
- `agent_type` (str): Type of agent
- `capabilities` (List[str]): List of agent capabilities
- `status` (AgentStatus): Current agent status
- `server_id` (str): ID of server hosting the agent
- `priority` (int): Agent priority level
- `resource_allocation` (Dict[str, float]): Current resource allocations
- `metadata` (Dict[str, Any]): Additional agent metadata

## Enumerations

### MCPServerStatus

Possible MCP server statuses.

```python
class MCPServerStatus(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTTING_DOWN = "shutting_down"
```

### AgentStatus

Possible agent statuses.

```python
class AgentStatus(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    BUSY = "busy"
```

### ErrorRecoveryStrategy

Available error recovery strategies.

```python
class ErrorRecoveryStrategy(Enum):
    RETRY = "retry"
    FAILOVER = "failover"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
```

## Error Handling

All SAFLA components implement comprehensive error handling with the following patterns:

### Exception Types

- `SAFLAError`: Base exception for all SAFLA-related errors
- `ValidationError`: Raised when validation fails
- `ConfigurationError`: Raised for configuration issues
- `MemoryError`: Raised for memory-related issues
- `SafetyViolationError`: Raised when safety constraints are violated

### Error Recovery

The framework provides automatic error recovery through:

1. **Retry Mechanisms**: Automatic retry with exponential backoff
2. **Failover**: Automatic failover to backup systems
3. **Circuit Breakers**: Prevent cascade failures
4. **Graceful Degradation**: Maintain partial functionality during failures

## Thread Safety

All SAFLA components are thread-safe and use appropriate locking mechanisms:

- `threading.Lock()` for critical sections
- `asyncio.Lock()` for async operations
- Atomic operations for simple state changes

## Performance Considerations

### Memory Management

- Automatic cleanup of expired contexts
- Memory consolidation for long-term storage
- Configurable memory limits and thresholds

### Scalability

- Asynchronous operations for I/O-bound tasks
- Load balancing across multiple servers
- Resource pooling and allocation

### Monitoring

- Built-in performance metrics collection
- Health monitoring for all components
- Comprehensive logging and debugging support

## Version Information

- **Current Version**: 0.1.0
- **API Stability**: Beta
- **Python Compatibility**: 3.8+
- **Dependencies**: See requirements.txt for full list

## See Also

- [SDK Guide](10-sdk-guide.md) - Integration patterns and best practices
- [Configuration Guide](11-configuration.md) - Detailed configuration options
- [Examples](12-examples.md) - Practical usage examples
- [Algorithms](13-algorithms.md) - Mathematical formulations and algorithms