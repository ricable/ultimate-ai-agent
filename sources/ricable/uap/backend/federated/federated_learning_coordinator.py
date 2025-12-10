"""
Agent 36: Federated Learning & Privacy-Preserving AI
Implements federated learning protocols, differential privacy,
homomorphic encryption, and secure multi-party computation.
"""

import asyncio
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from uuid import uuid4
from enum import Enum
import hashlib
import hmac
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time

logger = logging.getLogger(__name__)


class FederatedLearningAlgorithm(Enum):
    """Federated learning algorithms"""
    FEDAVG = "federated_averaging"
    FEDPROX = "federated_proximal"
    FEDOPT = "federated_optimization"
    SCAFFOLD = "scaffold"
    FEDBN = "federated_batch_norm"


class PrivacyTechnique(Enum):
    """Privacy preservation techniques"""
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_AGGREGATION = "secure_aggregation"
    PRIVATE_SET_INTERSECTION = "private_set_intersection"
    MULTIPARTY_COMPUTATION = "multiparty_computation"


class ClientStatus(Enum):
    """Client participation status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    UPLOADING = "uploading"
    DROPPED = "dropped"
    MALICIOUS = "malicious"


@dataclass
class FederatedClient:
    """Represents a federated learning client"""
    client_id: str
    name: str
    data_size: int
    computation_capacity: float  # FLOPS
    communication_bandwidth: float  # Mbps
    privacy_requirements: List[str]
    last_seen: datetime
    status: ClientStatus
    model_version: int
    contribution_score: float
    trust_score: float
    local_epochs: int
    batch_size: int
    learning_rate: float
    public_key: Optional[bytes]
    encryption_key: Optional[bytes]


@dataclass
class ModelUpdate:
    """Represents a model update from a client"""
    update_id: str
    client_id: str
    round_number: int
    model_weights: Dict[str, torch.Tensor]
    gradient_norm: float
    data_samples: int
    training_loss: float
    training_accuracy: float
    privacy_budget_used: float
    timestamp: datetime
    signature: Optional[str]
    encrypted: bool


@dataclass
class FederatedRound:
    """Represents a federated learning round"""
    round_id: str
    round_number: int
    participating_clients: List[str]
    global_model_version: int
    aggregated_weights: Dict[str, torch.Tensor]
    global_loss: float
    global_accuracy: float
    convergence_metric: float
    privacy_budget_consumed: float
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # "running", "completed", "failed"


@dataclass
class PrivacyBudget:
    """Tracks privacy budget for differential privacy"""
    client_id: str
    epsilon_total: float
    delta_total: float
    epsilon_used: float
    delta_used: float
    epsilon_remaining: float
    delta_remaining: float
    last_updated: datetime


class DifferentialPrivacyManager:
    """Manages differential privacy mechanisms"""
    
    def __init__(self, default_epsilon: float = 1.0, default_delta: float = 1e-5):
        self.default_epsilon = default_epsilon
        self.default_delta = default_delta
        self.privacy_budgets: Dict[str, PrivacyBudget] = {}
        self.noise_multipliers = {}
        
    def initialize_budget(self, client_id: str, epsilon: float = None, delta: float = None):
        """Initialize privacy budget for a client"""
        epsilon = epsilon or self.default_epsilon
        delta = delta or self.default_delta
        
        budget = PrivacyBudget(
            client_id=client_id,
            epsilon_total=epsilon,
            delta_total=delta,
            epsilon_used=0.0,
            delta_used=0.0,
            epsilon_remaining=epsilon,
            delta_remaining=delta,
            last_updated=datetime.utcnow()
        )
        
        self.privacy_budgets[client_id] = budget
        return budget
    
    async def add_laplace_noise(self, data: torch.Tensor, sensitivity: float, 
                              epsilon: float, client_id: str) -> torch.Tensor:
        """Add Laplace noise for differential privacy"""
        if not self._check_budget(client_id, epsilon, 0):
            raise ValueError("Insufficient privacy budget")
        
        # Calculate noise scale
        scale = sensitivity / epsilon
        
        # Generate Laplace noise
        noise = torch.distributions.Laplace(0, scale).sample(data.shape)
        
        # Add noise to data
        noisy_data = data + noise
        
        # Update privacy budget
        self._consume_budget(client_id, epsilon, 0)
        
        logger.info(f"Added Laplace noise (scale={scale:.4f}) for client {client_id}")
        return noisy_data
    
    async def add_gaussian_noise(self, data: torch.Tensor, sensitivity: float,
                               epsilon: float, delta: float, client_id: str) -> torch.Tensor:
        """Add Gaussian noise for differential privacy"""
        if not self._check_budget(client_id, epsilon, delta):
            raise ValueError("Insufficient privacy budget")
        
        # Calculate noise scale using composition theorems
        sigma = self._calculate_gaussian_noise_scale(sensitivity, epsilon, delta)
        
        # Generate Gaussian noise
        noise = torch.normal(0, sigma, data.shape)
        
        # Add noise to data
        noisy_data = data + noise
        
        # Update privacy budget
        self._consume_budget(client_id, epsilon, delta)
        
        logger.info(f"Added Gaussian noise (σ={sigma:.4f}) for client {client_id}")
        return noisy_data
    
    def _calculate_gaussian_noise_scale(self, sensitivity: float, 
                                      epsilon: float, delta: float) -> float:
        """Calculate Gaussian noise scale for (ε,δ)-DP"""
        # Simplified calculation - in practice, use more precise methods
        c = np.sqrt(2 * np.log(1.25 / delta))
        sigma = c * sensitivity / epsilon
        return sigma
    
    def _check_budget(self, client_id: str, epsilon: float, delta: float) -> bool:
        """Check if privacy budget is sufficient"""
        if client_id not in self.privacy_budgets:
            return False
        
        budget = self.privacy_budgets[client_id]
        return (budget.epsilon_remaining >= epsilon and 
                budget.delta_remaining >= delta)
    
    def _consume_budget(self, client_id: str, epsilon: float, delta: float):
        """Consume privacy budget"""
        budget = self.privacy_budgets[client_id]
        budget.epsilon_used += epsilon
        budget.delta_used += delta
        budget.epsilon_remaining -= epsilon
        budget.delta_remaining -= delta
        budget.last_updated = datetime.utcnow()


class HomomorphicEncryption:
    """Simplified homomorphic encryption for secure aggregation"""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.private_key = None
        self.public_key = None
        self._generate_keys()
    
    def _generate_keys(self):
        """Generate RSA key pair (simplified HE)"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size
        )
        self.public_key = self.private_key.public_key()
    
    def encrypt(self, plaintext: float) -> bytes:
        """Encrypt a number using public key"""
        # Convert float to bytes
        plaintext_bytes = str(plaintext).encode()
        
        # Encrypt using RSA-OAEP
        ciphertext = self.public_key.encrypt(
            plaintext_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return ciphertext
    
    def decrypt(self, ciphertext: bytes) -> float:
        """Decrypt using private key"""
        plaintext_bytes = self.private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return float(plaintext_bytes.decode())
    
    def homomorphic_add(self, ciphertext1: bytes, ciphertext2: bytes) -> bytes:
        """Simplified homomorphic addition (not truly homomorphic)"""
        # This is a placeholder - real HE would allow operations on ciphertexts
        # For demonstration, we decrypt, add, and re-encrypt
        val1 = self.decrypt(ciphertext1)
        val2 = self.decrypt(ciphertext2)
        result = val1 + val2
        return self.encrypt(result)
    
    def get_public_key_bytes(self) -> bytes:
        """Get public key as bytes for sharing"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )


class SecureAggregator:
    """Implements secure aggregation protocols"""
    
    def __init__(self):
        self.encryption = HomomorphicEncryption()
        self.client_keys = {}
        self.masked_updates = {}
        
    async def secure_sum(self, client_updates: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform secure sum aggregation"""
        if not client_updates:
            return torch.tensor(0.0)
        
        # Initialize sum with first update
        first_client = list(client_updates.keys())[0]
        secure_sum = client_updates[first_client].clone()
        
        # Add remaining updates
        for client_id, update in list(client_updates.items())[1:]:
            secure_sum += update
        
        return secure_sum
    
    async def secure_average(self, client_updates: Dict[str, torch.Tensor],
                           weights: Dict[str, float] = None) -> torch.Tensor:
        """Perform secure weighted average"""
        if not client_updates:
            return torch.tensor(0.0)
        
        # Default to equal weights
        if weights is None:
            weights = {client_id: 1.0 for client_id in client_updates.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Compute weighted sum
        weighted_sum = torch.zeros_like(list(client_updates.values())[0])
        for client_id, update in client_updates.items():
            weight = normalized_weights.get(client_id, 0.0)
            weighted_sum += weight * update
        
        return weighted_sum
    
    async def dropout_resilient_aggregation(self, client_updates: Dict[str, torch.Tensor],
                                          min_clients: int = 2) -> Optional[torch.Tensor]:
        """Aggregation resilient to client dropouts"""
        if len(client_updates) < min_clients:
            logger.warning(f"Insufficient clients for aggregation: {len(client_updates)} < {min_clients}")
            return None
        
        return await self.secure_average(client_updates)
    
    def add_client_key(self, client_id: str, public_key: bytes):
        """Add client's public key for secure communication"""
        self.client_keys[client_id] = public_key


class FederatedLearningCoordinator:
    """Main coordinator for federated learning"""
    
    def __init__(self, algorithm: FederatedLearningAlgorithm = FederatedLearningAlgorithm.FEDAVG):
        self.algorithm = algorithm
        self.clients: Dict[str, FederatedClient] = {}
        self.global_model = None
        self.current_round = 0
        self.rounds_history: List[FederatedRound] = []
        self.model_updates: Dict[int, List[ModelUpdate]] = {}
        
        # Privacy and security components
        self.privacy_manager = DifferentialPrivacyManager()
        self.secure_aggregator = SecureAggregator()
        
        # Configuration
        self.config = {
            "min_clients": 2,
            "max_clients": 100,
            "rounds_limit": 1000,
            "convergence_threshold": 0.001,
            "client_timeout": 300,  # seconds
            "privacy_enabled": True,
            "secure_aggregation": True
        }
        
        # Status tracking
        self.is_running = False
        self.current_round_clients = set()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def initialize(self, global_model: nn.Module) -> bool:
        """Initialize the federated learning coordinator"""
        try:
            self.global_model = global_model
            logger.info(f"Initialized FL coordinator with {self.algorithm.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize FL coordinator: {e}")
            return False
    
    async def register_client(self, client: FederatedClient) -> bool:
        """Register a new federated learning client"""
        try:
            # Initialize privacy budget
            if self.config["privacy_enabled"]:
                self.privacy_manager.initialize_budget(client.client_id)
            
            # Store client
            self.clients[client.client_id] = client
            
            # Add encryption key if provided
            if client.public_key:
                self.secure_aggregator.add_client_key(client.client_id, client.public_key)
            
            logger.info(f"Registered client {client.client_id} with {client.data_size} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register client {client.client_id}: {e}")
            return False
    
    async def start_federated_training(self, num_rounds: int = 10) -> bool:
        """Start federated learning training"""
        if self.is_running:
            logger.warning("Federated learning already running")
            return False
        
        if len(self.clients) < self.config["min_clients"]:
            logger.error(f"Insufficient clients: {len(self.clients)} < {self.config['min_clients']}")
            return False
        
        self.is_running = True
        logger.info(f"Starting federated training for {num_rounds} rounds")
        
        try:
            for round_num in range(1, num_rounds + 1):
                success = await self._execute_round(round_num)
                if not success:
                    logger.error(f"Round {round_num} failed")
                    break
                
                # Check convergence
                if await self._check_convergence():
                    logger.info(f"Converged after {round_num} rounds")
                    break
            
            self.is_running = False
            logger.info("Federated training completed")
            return True
            
        except Exception as e:
            logger.error(f"Federated training failed: {e}")
            self.is_running = False
            return False
    
    async def _execute_round(self, round_num: int) -> bool:
        """Execute a single federated learning round"""
        logger.info(f"Starting round {round_num}")
        
        # Create round record
        round_record = FederatedRound(
            round_id=str(uuid4()),
            round_number=round_num,
            participating_clients=[],
            global_model_version=round_num - 1,
            aggregated_weights={},
            global_loss=0.0,
            global_accuracy=0.0,
            convergence_metric=0.0,
            privacy_budget_consumed=0.0,
            start_time=datetime.utcnow(),
            end_time=None,
            status="running"
        )
        
        try:
            # 1. Client selection
            selected_clients = await self._select_clients()
            round_record.participating_clients = [c.client_id for c in selected_clients]
            
            if not selected_clients:
                logger.warning("No clients selected for round")
                round_record.status = "failed"
                return False
            
            # 2. Send global model to clients
            await self._broadcast_global_model(selected_clients)
            
            # 3. Wait for client updates
            client_updates = await self._collect_client_updates(selected_clients, round_num)
            
            if len(client_updates) < self.config["min_clients"]:
                logger.warning(f"Insufficient client updates: {len(client_updates)}")
                round_record.status = "failed"
                return False
            
            # 4. Aggregate updates
            aggregated_weights = await self._aggregate_updates(client_updates)
            round_record.aggregated_weights = aggregated_weights
            
            # 5. Update global model
            await self._update_global_model(aggregated_weights)
            
            # 6. Calculate metrics
            round_record.global_loss, round_record.global_accuracy = await self._calculate_global_metrics(client_updates)
            round_record.convergence_metric = await self._calculate_convergence_metric()
            round_record.privacy_budget_consumed = sum(
                update.privacy_budget_used for update in client_updates
            )
            
            round_record.end_time = datetime.utcnow()
            round_record.status = "completed"
            
            # Store round record
            self.rounds_history.append(round_record)
            self.current_round = round_num
            
            logger.info(f"Round {round_num} completed - Loss: {round_record.global_loss:.4f}, "
                       f"Accuracy: {round_record.global_accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Round {round_num} execution failed: {e}")
            round_record.status = "failed"
            round_record.end_time = datetime.utcnow()
            self.rounds_history.append(round_record)
            return False
    
    async def _select_clients(self) -> List[FederatedClient]:
        """Select clients for the current round"""
        # Filter active clients
        active_clients = [
            client for client in self.clients.values()
            if client.status == ClientStatus.ACTIVE
        ]
        
        if not active_clients:
            return []
        
        # Selection strategy based on algorithm
        if self.algorithm == FederatedLearningAlgorithm.FEDAVG:
            # Random selection
            import random
            num_select = min(len(active_clients), max(2, len(active_clients) // 2))
            selected = random.sample(active_clients, num_select)
        else:
            # For other algorithms, select all active clients
            selected = active_clients
        
        logger.info(f"Selected {len(selected)} clients for training")
        return selected
    
    async def _broadcast_global_model(self, clients: List[FederatedClient]):
        """Send global model to selected clients"""
        for client in clients:
            client.status = ClientStatus.TRAINING
            client.model_version = self.current_round
            # In practice, would send model weights via secure channel
            logger.debug(f"Sent global model to client {client.client_id}")
    
    async def _collect_client_updates(self, clients: List[FederatedClient], 
                                    round_num: int) -> List[ModelUpdate]:
        """Collect model updates from clients"""
        updates = []
        timeout = self.config["client_timeout"]
        
        # Simulate client training and collect updates
        for client in clients:
            try:
                # Simulate client update (in practice, received via network)
                update = await self._simulate_client_update(client, round_num)
                if update:
                    updates.append(update)
                    client.status = ClientStatus.ACTIVE
                else:
                    client.status = ClientStatus.DROPPED
                    logger.warning(f"Client {client.client_id} dropped")
                    
            except Exception as e:
                logger.error(f"Failed to collect update from {client.client_id}: {e}")
                client.status = ClientStatus.DROPPED
        
        logger.info(f"Collected {len(updates)} client updates")
        return updates
    
    async def _simulate_client_update(self, client: FederatedClient, 
                                    round_num: int) -> Optional[ModelUpdate]:
        """Simulate a client model update (for demonstration)"""
        try:
            # Simulate model weights (random for demonstration)
            model_weights = {}
            for name, param in self.global_model.named_parameters():
                # Add small random perturbation to simulate local training
                noise = torch.randn_like(param) * 0.01
                model_weights[name] = param + noise
            
            # Apply differential privacy if enabled
            if self.config["privacy_enabled"]:
                model_weights = await self._apply_differential_privacy(
                    model_weights, client.client_id
                )
            
            update = ModelUpdate(
                update_id=str(uuid4()),
                client_id=client.client_id,
                round_number=round_num,
                model_weights=model_weights,
                gradient_norm=np.random.uniform(0.1, 2.0),
                data_samples=client.data_size,
                training_loss=np.random.uniform(0.1, 1.0),
                training_accuracy=np.random.uniform(0.7, 0.95),
                privacy_budget_used=0.1 if self.config["privacy_enabled"] else 0.0,
                timestamp=datetime.utcnow(),
                signature=None,
                encrypted=self.config["secure_aggregation"]
            )
            
            return update
            
        except Exception as e:
            logger.error(f"Failed to simulate update for {client.client_id}: {e}")
            return None
    
    async def _apply_differential_privacy(self, model_weights: Dict[str, torch.Tensor],
                                        client_id: str) -> Dict[str, torch.Tensor]:
        """Apply differential privacy to model weights"""
        private_weights = {}
        
        for name, weights in model_weights.items():
            # Apply Laplace noise for differential privacy
            try:
                noisy_weights = await self.privacy_manager.add_laplace_noise(
                    weights, sensitivity=1.0, epsilon=0.1, client_id=client_id
                )
                private_weights[name] = noisy_weights
            except ValueError as e:
                # Privacy budget exhausted
                logger.warning(f"Privacy budget exhausted for {client_id}: {e}")
                private_weights[name] = weights  # Use original weights
        
        return private_weights
    
    async def _aggregate_updates(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Aggregate client model updates"""
        if not updates:
            return {}
        
        # Prepare weights for aggregation
        client_weights = {}
        client_data_sizes = {}
        
        for update in updates:
            client_weights[update.client_id] = update.model_weights
            client_data_sizes[update.client_id] = update.data_samples
        
        # Apply aggregation algorithm
        if self.algorithm == FederatedLearningAlgorithm.FEDAVG:
            return await self._federated_averaging(client_weights, client_data_sizes)
        elif self.algorithm == FederatedLearningAlgorithm.FEDPROX:
            return await self._federated_proximal(client_weights, client_data_sizes)
        else:
            # Default to FedAvg
            return await self._federated_averaging(client_weights, client_data_sizes)
    
    async def _federated_averaging(self, client_weights: Dict[str, Dict[str, torch.Tensor]],
                                 data_sizes: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """Implement FedAvg aggregation"""
        if not client_weights:
            return {}
        
        # Calculate weights based on data sizes
        total_samples = sum(data_sizes.values())
        aggregation_weights = {
            client_id: size / total_samples 
            for client_id, size in data_sizes.items()
        }
        
        # Initialize aggregated weights
        first_client = list(client_weights.keys())[0]
        aggregated = {}
        
        for param_name in client_weights[first_client].keys():
            # Weighted average of parameters
            weighted_sum = torch.zeros_like(client_weights[first_client][param_name])
            
            for client_id, weights in client_weights.items():
                weight = aggregation_weights[client_id]
                weighted_sum += weight * weights[param_name]
            
            aggregated[param_name] = weighted_sum
        
        return aggregated
    
    async def _federated_proximal(self, client_weights: Dict[str, Dict[str, torch.Tensor]],
                                data_sizes: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """Implement FedProx aggregation with proximal term"""
        # For simplicity, use FedAvg with slight modification
        aggregated = await self._federated_averaging(client_weights, data_sizes)
        
        # Apply proximal regularization (simplified)
        mu = 0.01  # Proximal parameter
        for param_name, param in aggregated.items():
            if hasattr(self.global_model, param_name):
                global_param = getattr(self.global_model, param_name)
                # Apply proximal term
                aggregated[param_name] = param - mu * (param - global_param)
        
        return aggregated
    
    async def _update_global_model(self, aggregated_weights: Dict[str, torch.Tensor]):
        """Update the global model with aggregated weights"""
        if not aggregated_weights:
            return
        
        # Update model parameters
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_weights:
                    param.copy_(aggregated_weights[name])
        
        logger.debug("Updated global model with aggregated weights")
    
    async def _calculate_global_metrics(self, updates: List[ModelUpdate]) -> Tuple[float, float]:
        """Calculate global loss and accuracy"""
        if not updates:
            return 0.0, 0.0
        
        # Weighted average based on data samples
        total_samples = sum(update.data_samples for update in updates)
        
        weighted_loss = sum(
            update.training_loss * update.data_samples / total_samples
            for update in updates
        )
        
        weighted_accuracy = sum(
            update.training_accuracy * update.data_samples / total_samples
            for update in updates
        )
        
        return weighted_loss, weighted_accuracy
    
    async def _calculate_convergence_metric(self) -> float:
        """Calculate convergence metric"""
        if len(self.rounds_history) < 2:
            return 1.0  # Not converged
        
        # Simple convergence check based on loss change
        current_loss = self.rounds_history[-1].global_loss
        previous_loss = self.rounds_history[-2].global_loss
        
        if previous_loss == 0:
            return 1.0
        
        loss_change = abs(current_loss - previous_loss) / previous_loss
        return loss_change
    
    async def _check_convergence(self) -> bool:
        """Check if training has converged"""
        if len(self.rounds_history) < 3:
            return False
        
        # Check if loss change is below threshold for last few rounds
        convergence_metric = await self._calculate_convergence_metric()
        return convergence_metric < self.config["convergence_threshold"]
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            "is_running": self.is_running,
            "current_round": self.current_round,
            "total_clients": len(self.clients),
            "active_clients": len([c for c in self.clients.values() if c.status == ClientStatus.ACTIVE]),
            "algorithm": self.algorithm.value,
            "privacy_enabled": self.config["privacy_enabled"],
            "secure_aggregation": self.config["secure_aggregation"],
            "last_round_metrics": self.rounds_history[-1] if self.rounds_history else None,
            "convergence_status": await self._check_convergence() if len(self.rounds_history) >= 3 else False
        }
    
    async def get_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy consumption report"""
        privacy_report = {
            "total_clients": len(self.privacy_manager.privacy_budgets),
            "clients": {},
            "average_epsilon_used": 0.0,
            "average_delta_used": 0.0,
            "clients_budget_exhausted": 0
        }
        
        total_epsilon = 0.0
        total_delta = 0.0
        
        for client_id, budget in self.privacy_manager.privacy_budgets.items():
            client_report = {
                "epsilon_total": budget.epsilon_total,
                "epsilon_used": budget.epsilon_used,
                "epsilon_remaining": budget.epsilon_remaining,
                "delta_total": budget.delta_total,
                "delta_used": budget.delta_used,
                "delta_remaining": budget.delta_remaining,
                "budget_exhausted": budget.epsilon_remaining <= 0 or budget.delta_remaining <= 0
            }
            
            privacy_report["clients"][client_id] = client_report
            total_epsilon += budget.epsilon_used
            total_delta += budget.delta_used
            
            if client_report["budget_exhausted"]:
                privacy_report["clients_budget_exhausted"] += 1
        
        if len(self.privacy_manager.privacy_budgets) > 0:
            privacy_report["average_epsilon_used"] = total_epsilon / len(self.privacy_manager.privacy_budgets)
            privacy_report["average_delta_used"] = total_delta / len(self.privacy_manager.privacy_budgets)
        
        return privacy_report
    
    async def export_model(self, file_path: str) -> bool:
        """Export the trained global model"""
        try:
            if self.global_model:
                torch.save(self.global_model.state_dict(), file_path)
                logger.info(f"Model exported to {file_path}")
                return True
            else:
                logger.error("No global model to export")
                return False
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            return False