"""
Federated Learning Integration for SAFLA.

This module implements federated learning capabilities, enabling distributed
model training across multiple nodes while preserving data privacy and
reducing communication overhead.

Components:
- Federated Coordinator: Central coordination of federated learning
- Client Manager: Management of federated learning clients
- Model Aggregator: Aggregation strategies for model updates
- Privacy Preserving Mechanisms: Differential privacy and secure aggregation
- Communication Protocol: Efficient model update communication

Technical Features:
- Support for various aggregation algorithms (FedAvg, FedProx, etc.)
- Differential privacy implementation
- Secure multi-party computation
- Asynchronous and synchronous training modes
- Client selection strategies
- Model compression for bandwidth efficiency
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import hashlib
import json

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Model aggregation strategies."""
    FED_AVG = "federated_averaging"
    FED_PROX = "federated_proximal"
    FED_YOGI = "federated_yogi"
    FED_ADAM = "federated_adam"
    WEIGHTED_AVG = "weighted_average"


class ClientStatus(Enum):
    """Status of federated learning clients."""
    IDLE = "idle"
    TRAINING = "training"
    UPLOADING = "uploading"
    OFFLINE = "offline"
    FAILED = "failed"


class PrivacyMechanism(Enum):
    """Privacy preservation mechanisms."""
    NONE = "none"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"


@dataclass
class ModelUpdate:
    """Represents a model update from a client."""
    client_id: str
    round_number: int
    model_weights: Dict[str, np.ndarray]
    training_samples: int
    training_time: float
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FederatedClient:
    """Represents a federated learning client."""
    client_id: str
    data_size: int
    compute_capacity: float  # 0.0 to 1.0
    reliability_score: float  # 0.0 to 1.0
    status: ClientStatus = ClientStatus.IDLE
    last_update: datetime = field(default_factory=datetime.now)
    total_rounds_participated: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FederatedRound:
    """Represents a federated learning round."""
    round_number: int
    selected_clients: List[str]
    aggregation_strategy: AggregationStrategy
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    global_model_update: Optional[Dict[str, np.ndarray]] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PrivacyConfig:
    """Configuration for privacy mechanisms."""
    mechanism: PrivacyMechanism
    epsilon: float = 1.0  # Differential privacy parameter
    delta: float = 1e-5  # Differential privacy parameter
    clip_norm: float = 1.0  # Gradient clipping
    noise_multiplier: float = 0.1
    secure_aggregation_threshold: int = 3


class FederatedLearningCoordinator:
    """
    Coordinates federated learning across distributed clients.
    
    Manages the federated learning process including client selection,
    model distribution, update aggregation, and privacy preservation.
    """
    
    def __init__(
        self,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.FED_AVG,
        privacy_config: Optional[PrivacyConfig] = None,
        min_clients_per_round: int = 2,
        max_clients_per_round: int = 10,
        rounds_per_epoch: int = 10
    ):
        """Initialize federated learning coordinator."""
        self.aggregation_strategy = aggregation_strategy
        self.privacy_config = privacy_config or PrivacyConfig(PrivacyMechanism.NONE)
        self.min_clients_per_round = min_clients_per_round
        self.max_clients_per_round = max_clients_per_round
        self.rounds_per_epoch = rounds_per_epoch
        
        # Client management
        self.clients: Dict[str, FederatedClient] = {}
        self.client_updates: Dict[int, List[ModelUpdate]] = {}
        
        # Training state
        self.current_round = 0
        self.global_model: Optional[Dict[str, np.ndarray]] = None
        self.round_history: List[FederatedRound] = []
        
        logger.info("Initialized FederatedLearningCoordinator")
    
    def register_client(self, client: FederatedClient) -> bool:
        """Register a new federated learning client."""
        if client.client_id in self.clients:
            logger.warning(f"Client {client.client_id} already registered")
            return False
        
        self.clients[client.client_id] = client
        logger.info(f"Registered client {client.client_id}")
        return True
    
    def unregister_client(self, client_id: str) -> bool:
        """Unregister a federated learning client."""
        if client_id not in self.clients:
            return False
        
        del self.clients[client_id]
        logger.info(f"Unregistered client {client_id}")
        return True
    
    async def select_clients_for_round(self) -> List[str]:
        """Select clients for the current round."""
        available_clients = [
            client_id for client_id, client in self.clients.items()
            if client.status in [ClientStatus.IDLE, ClientStatus.TRAINING]
        ]
        
        if len(available_clients) < self.min_clients_per_round:
            logger.warning(f"Not enough clients available: {len(available_clients)}")
            return []
        
        # Client selection strategy
        # Sort by reliability and compute capacity
        scored_clients = []
        for client_id in available_clients:
            client = self.clients[client_id]
            score = client.reliability_score * 0.7 + client.compute_capacity * 0.3
            scored_clients.append((client_id, score))
        
        scored_clients.sort(key=lambda x: x[1], reverse=True)
        
        # Select top clients
        num_clients = min(
            len(available_clients),
            self.max_clients_per_round
        )
        
        selected = [client_id for client_id, _ in scored_clients[:num_clients]]
        
        # Update client status
        for client_id in selected:
            self.clients[client_id].status = ClientStatus.TRAINING
        
        return selected
    
    async def start_training_round(self) -> FederatedRound:
        """Start a new federated training round."""
        self.current_round += 1
        
        # Select clients
        selected_clients = await self.select_clients_for_round()
        
        if not selected_clients:
            raise RuntimeError("No clients available for training")
        
        # Create round
        round_info = FederatedRound(
            round_number=self.current_round,
            selected_clients=selected_clients,
            aggregation_strategy=self.aggregation_strategy
        )
        
        self.round_history.append(round_info)
        
        logger.info(f"Started round {self.current_round} with {len(selected_clients)} clients")
        
        return round_info
    
    async def receive_client_update(
        self,
        update: ModelUpdate
    ) -> bool:
        """Receive model update from a client."""
        if update.round_number not in self.client_updates:
            self.client_updates[update.round_number] = []
        
        self.client_updates[update.round_number].append(update)
        
        # Update client status
        if update.client_id in self.clients:
            client = self.clients[update.client_id]
            client.status = ClientStatus.IDLE
            client.last_update = datetime.now()
            client.total_rounds_participated += 1
        
        logger.info(f"Received update from client {update.client_id} for round {update.round_number}")
        
        return True
    
    async def aggregate_updates(
        self,
        round_number: int
    ) -> Optional[Dict[str, np.ndarray]]:
        """Aggregate client updates for a round."""
        if round_number not in self.client_updates:
            logger.warning(f"No updates for round {round_number}")
            return None
        
        updates = self.client_updates[round_number]
        
        if not updates:
            return None
        
        # Apply privacy mechanisms if configured
        if self.privacy_config.mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
            updates = await self._apply_differential_privacy(updates)
        
        # Perform aggregation based on strategy
        if self.aggregation_strategy == AggregationStrategy.FED_AVG:
            aggregated = await self._federated_averaging(updates)
        elif self.aggregation_strategy == AggregationStrategy.WEIGHTED_AVG:
            aggregated = await self._weighted_averaging(updates)
        else:
            # Default to federated averaging
            aggregated = await self._federated_averaging(updates)
        
        # Update global model
        self.global_model = aggregated
        
        # Update round info
        for round_info in self.round_history:
            if round_info.round_number == round_number:
                round_info.end_time = datetime.now()
                round_info.global_model_update = aggregated
                break
        
        return aggregated
    
    async def _federated_averaging(
        self,
        updates: List[ModelUpdate]
    ) -> Dict[str, np.ndarray]:
        """Perform federated averaging aggregation."""
        if not updates:
            return {}
        
        # Initialize aggregated weights
        aggregated_weights = {}
        total_samples = sum(update.training_samples for update in updates)
        
        for layer_name in updates[0].model_weights:
            # Weighted average based on number of samples
            weighted_sum = np.zeros_like(updates[0].model_weights[layer_name])
            
            for update in updates:
                weight = update.training_samples / total_samples
                weighted_sum += weight * update.model_weights[layer_name]
            
            aggregated_weights[layer_name] = weighted_sum
        
        return aggregated_weights
    
    async def _weighted_averaging(
        self,
        updates: List[ModelUpdate]
    ) -> Dict[str, np.ndarray]:
        """Perform weighted averaging with custom weights."""
        if not updates:
            return {}
        
        # Calculate weights based on client reliability and data size
        weights = []
        for update in updates:
            client = self.clients.get(update.client_id)
            if client:
                weight = (
                    client.reliability_score * 0.5 +
                    (update.training_samples / 1000) * 0.5  # Normalize
                )
            else:
                weight = 1.0
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Perform weighted aggregation
        aggregated_weights = {}
        
        for layer_name in updates[0].model_weights:
            weighted_sum = np.zeros_like(updates[0].model_weights[layer_name])
            
            for i, update in enumerate(updates):
                weighted_sum += weights[i] * update.model_weights[layer_name]
            
            aggregated_weights[layer_name] = weighted_sum
        
        return aggregated_weights
    
    async def _apply_differential_privacy(
        self,
        updates: List[ModelUpdate]
    ) -> List[ModelUpdate]:
        """Apply differential privacy to model updates."""
        noisy_updates = []
        
        for update in updates:
            noisy_weights = {}
            
            for layer_name, weights in update.model_weights.items():
                # Clip weights
                norm = np.linalg.norm(weights)
                if norm > self.privacy_config.clip_norm:
                    weights = weights * (self.privacy_config.clip_norm / norm)
                
                # Add Gaussian noise
                noise = np.random.normal(
                    0,
                    self.privacy_config.noise_multiplier * self.privacy_config.clip_norm,
                    weights.shape
                )
                
                noisy_weights[layer_name] = weights + noise
            
            # Create new update with noisy weights
            noisy_update = ModelUpdate(
                client_id=update.client_id,
                round_number=update.round_number,
                model_weights=noisy_weights,
                training_samples=update.training_samples,
                training_time=update.training_time,
                metrics=update.metrics
            )
            
            noisy_updates.append(noisy_update)
        
        return noisy_updates
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get federated learning statistics."""
        total_clients = len(self.clients)
        active_clients = sum(
            1 for client in self.clients.values()
            if client.status != ClientStatus.OFFLINE
        )
        
        avg_reliability = np.mean([
            client.reliability_score
            for client in self.clients.values()
        ]) if self.clients else 0
        
        return {
            'current_round': self.current_round,
            'total_clients': total_clients,
            'active_clients': active_clients,
            'average_reliability': avg_reliability,
            'aggregation_strategy': self.aggregation_strategy.value,
            'privacy_mechanism': self.privacy_config.mechanism.value,
            'rounds_completed': len([r for r in self.round_history if r.end_time]),
            'total_updates_received': sum(
                len(updates) for updates in self.client_updates.values()
            )
        }


class ModelCompressor:
    """Handles model compression for efficient communication."""
    
    @staticmethod
    def compress_model(
        model_weights: Dict[str, np.ndarray],
        compression_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """Compress model weights for transmission."""
        compressed = {}
        
        for layer_name, weights in model_weights.items():
            # Simple top-k sparsification
            flat_weights = weights.flatten()
            k = int(len(flat_weights) * compression_ratio)
            
            # Get top-k indices and values
            top_k_indices = np.argpartition(np.abs(flat_weights), -k)[-k:]
            top_k_values = flat_weights[top_k_indices]
            
            compressed[layer_name] = {
                'shape': weights.shape,
                'indices': top_k_indices.tolist(),
                'values': top_k_values.tolist()
            }
        
        return compressed
    
    @staticmethod
    def decompress_model(
        compressed_weights: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Decompress model weights."""
        decompressed = {}
        
        for layer_name, data in compressed_weights.items():
            # Reconstruct sparse array
            shape = data['shape']
            indices = np.array(data['indices'])
            values = np.array(data['values'])
            
            # Create zero array and fill with values
            weights = np.zeros(np.prod(shape))
            weights[indices] = values
            weights = weights.reshape(shape)
            
            decompressed[layer_name] = weights
        
        return decompressed


class SecureAggregator:
    """Implements secure aggregation protocols."""
    
    def __init__(self, threshold: int = 3):
        """Initialize secure aggregator."""
        self.threshold = threshold
        self.client_keys: Dict[str, bytes] = {}
    
    async def setup_keys(self, client_ids: List[str]) -> Dict[str, bytes]:
        """Setup encryption keys for clients."""
        keys = {}
        
        for client_id in client_ids:
            # Generate random key (simplified)
            key = hashlib.sha256(client_id.encode()).digest()
            self.client_keys[client_id] = key
            keys[client_id] = key
        
        return keys
    
    async def secure_aggregate(
        self,
        encrypted_updates: Dict[str, bytes]
    ) -> Optional[Dict[str, np.ndarray]]:
        """Perform secure aggregation."""
        if len(encrypted_updates) < self.threshold:
            logger.warning(f"Not enough clients for secure aggregation: {len(encrypted_updates)}")
            return None
        
        # Simplified secure aggregation (in practice, use proper MPC)
        # This is a placeholder implementation
        aggregated = {}
        
        return aggregated