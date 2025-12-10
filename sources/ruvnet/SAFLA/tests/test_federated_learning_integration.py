"""
Test suite for Federated Learning Integration - Distributed knowledge sharing without raw data transfer.

This module tests the federated learning capabilities for SAFLA:
- Federated model training and aggregation
- Privacy-preserving knowledge sharing
- Differential privacy mechanisms
- Secure multi-party computation
- Model versioning and synchronization

Following TDD principles: Red-Green-Refactor cycle.
"""

import pytest
import numpy as np
import torch
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import dataclass, field
from enum import Enum

# Import the classes we'll implement
from safla.core.federated_learning_integration import (
    FederatedLearningCoordinator,
    FederatedClient,
    AggregationStrategy,
    ClientStatus,
    PrivacyMechanism,
    ModelUpdate,
    FederatedRound,
    PrivacyConfig,
    ModelCompressor,
    SecureAggregator
)


class ClientStatus(Enum):
    """Client status enumeration."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    TRAINING = "training"
    UPLOADING = "uploading"
    FAILED = "failed"
    DISCONNECTED = "disconnected"


@dataclass
class MockFederatedClient:
    """Mock federated client for testing."""
    client_id: str
    data_size: int
    model_version: str = "v1.0"
    status: ClientStatus = ClientStatus.INACTIVE
    privacy_budget: float = 1.0
    last_update: datetime = field(default_factory=datetime.now)
    capabilities: List[str] = field(default_factory=lambda: ["training", "inference"])


class TestFederatedLearningCoordinator:
    """Test suite for Federated Learning Coordinator functionality."""
    
    @pytest.fixture
    def federated_config(self):
        """Create federated learning configuration for testing."""
        return FederatedConfig(
            aggregation_strategy="fedavg",
            min_clients_per_round=3,
            max_clients_per_round=10,
            rounds_per_epoch=5,
            client_fraction=0.6,
            privacy_enabled=True,
            differential_privacy_epsilon=1.0,
            differential_privacy_delta=1e-5,
            secure_aggregation=True,
            model_compression=True,
            communication_rounds=100,
            convergence_threshold=0.001
        )
    
    @pytest.fixture
    def coordinator(self, federated_config):
        """Create FederatedLearningCoordinator instance for testing."""
        return FederatedLearningCoordinator(config=federated_config)
    
    @pytest.fixture
    def mock_clients(self):
        """Create mock federated clients for testing."""
        return [
            MockFederatedClient(
                client_id=f"client_{i}",
                data_size=1000 + (i * 500),
                model_version="v1.0",
                privacy_budget=1.0 - (i * 0.1)
            )
            for i in range(8)
        ]
    
    def test_coordinator_initialization(self, coordinator, federated_config):
        """Test FederatedLearningCoordinator initialization."""
        assert coordinator.config == federated_config
        assert coordinator.model_aggregator is not None
        assert coordinator.privacy_engine is not None
        assert coordinator.version_manager is not None
        assert coordinator.current_round == 0
        assert coordinator.global_model is not None
        assert coordinator.is_training is False
    
    @pytest.mark.asyncio
    async def test_client_registration(self, coordinator, mock_clients):
        """Test client registration and management."""
        client = mock_clients[0]
        
        # Register client
        success = await coordinator.register_client(client)
        assert success
        assert coordinator.get_client_count() == 1
        assert coordinator.has_client(client.client_id)
        
        # Register multiple clients
        for client in mock_clients[1:4]:
            await coordinator.register_client(client)
        
        assert coordinator.get_client_count() == 4
        
        # Test client deregistration
        success = await coordinator.deregister_client(mock_clients[0].client_id)
        assert success
        assert coordinator.get_client_count() == 3
        assert not coordinator.has_client(mock_clients[0].client_id)
    
    @pytest.mark.asyncio
    async def test_client_selection_for_training(self, coordinator, mock_clients):
        """Test client selection for federated training rounds."""
        # Register clients
        for client in mock_clients:
            await coordinator.register_client(client)
        
        # Select clients for training round
        selected_clients = coordinator.select_clients_for_round()
        
        # Should select between min and max clients
        assert coordinator.config.min_clients_per_round <= len(selected_clients) <= coordinator.config.max_clients_per_round
        
        # Should respect client fraction
        expected_count = int(len(mock_clients) * coordinator.config.client_fraction)
        assert len(selected_clients) <= expected_count
        
        # All selected clients should be registered
        for client_id in selected_clients:
            assert coordinator.has_client(client_id)
    
    @pytest.mark.asyncio
    async def test_federated_training_round(self, coordinator, mock_clients):
        """Test complete federated training round."""
        # Register clients
        for client in mock_clients[:5]:
            await coordinator.register_client(client)
        
        # Mock client training
        with patch.object(coordinator, '_train_client_model') as mock_train:
            mock_train.return_value = {
                "model_update": torch.randn(100),  # Mock model weights
                "num_samples": 1000,
                "loss": 0.5,
                "accuracy": 0.85
            }
            
            # Execute training round
            round_results = await coordinator.execute_training_round()
            
            assert "participating_clients" in round_results
            assert "aggregation_result" in round_results
            assert "round_metrics" in round_results
            
            # Should have called training for selected clients
            assert mock_train.call_count >= coordinator.config.min_clients_per_round
    
    @pytest.mark.asyncio
    async def test_model_aggregation(self, coordinator, mock_clients):
        """Test federated model aggregation."""
        # Register clients
        for client in mock_clients[:4]:
            await coordinator.register_client(client)
        
        # Create mock client updates
        client_updates = []
        for i, client in enumerate(mock_clients[:4]):
            update = {
                "client_id": client.client_id,
                "model_weights": torch.randn(100),
                "num_samples": 1000 + (i * 200),
                "training_loss": 0.5 - (i * 0.05)
            }
            client_updates.append(update)
        
        # Aggregate models
        aggregation_result = await coordinator.aggregate_models(client_updates)
        
        assert aggregation_result.success
        assert aggregation_result.aggregated_weights is not None
        assert aggregation_result.participating_clients == 4
        assert aggregation_result.total_samples == sum(update["num_samples"] for update in client_updates)
    
    def test_privacy_budget_management(self, coordinator, mock_clients):
        """Test privacy budget tracking and management."""
        # Register clients with different privacy budgets
        for client in mock_clients[:3]:
            coordinator.register_client(client)
        
        privacy_engine = coordinator.privacy_engine
        
        # Check initial privacy budgets
        for client in mock_clients[:3]:
            budget = privacy_engine.get_privacy_budget(client.client_id)
            assert budget.epsilon > 0
            assert budget.delta > 0
        
        # Consume privacy budget
        consumption = PrivacyBudget(epsilon=0.1, delta=1e-6)
        success = privacy_engine.consume_budget(mock_clients[0].client_id, consumption)
        assert success
        
        # Check updated budget
        updated_budget = privacy_engine.get_privacy_budget(mock_clients[0].client_id)
        assert updated_budget.epsilon < mock_clients[0].privacy_budget
    
    @pytest.mark.asyncio
    async def test_differential_privacy_mechanism(self, coordinator):
        """Test differential privacy noise addition."""
        privacy_engine = coordinator.privacy_engine
        dp_mechanism = privacy_engine.differential_privacy
        
        # Test noise addition to gradients
        original_gradients = torch.randn(50)
        epsilon = 1.0
        delta = 1e-5
        sensitivity = 1.0
        
        noisy_gradients = dp_mechanism.add_noise(
            gradients=original_gradients,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity
        )
        
        assert noisy_gradients.shape == original_gradients.shape
        assert not torch.equal(noisy_gradients, original_gradients)
        
        # Test privacy accounting
        privacy_cost = dp_mechanism.compute_privacy_cost(
            epsilon=epsilon,
            delta=delta,
            num_steps=1
        )
        
        assert privacy_cost.epsilon <= epsilon
        assert privacy_cost.delta <= delta
    
    @pytest.mark.asyncio
    async def test_secure_aggregation_protocol(self, coordinator, mock_clients):
        """Test secure multi-party aggregation."""
        # Register clients
        for client in mock_clients[:4]:
            await coordinator.register_client(client)
        
        secure_agg = coordinator.privacy_engine.secure_aggregation
        
        # Create client contributions
        client_contributions = {}
        for client in mock_clients[:4]:
            # Mock encrypted model update
            contribution = {
                "encrypted_weights": torch.randn(100),
                "commitment": hashlib.sha256(f"client_{client.client_id}".encode()).hexdigest(),
                "proof": "mock_zero_knowledge_proof"
            }
            client_contributions[client.client_id] = contribution
        
        # Perform secure aggregation
        with patch.object(secure_agg, '_decrypt_and_aggregate') as mock_decrypt:
            mock_decrypt.return_value = torch.randn(100)
            
            aggregated_result = await secure_agg.secure_aggregate(client_contributions)
            
            assert aggregated_result is not None
            assert aggregated_result.shape == (100,)
            mock_decrypt.assert_called_once()
    
    def test_model_compression_and_quantization(self, coordinator):
        """Test model compression for efficient communication."""
        # Create mock model weights
        model_weights = torch.randn(1000)
        
        # Test compression
        compressed_weights = coordinator.compress_model(model_weights)
        
        assert compressed_weights.numel() <= model_weights.numel()
        
        # Test decompression
        decompressed_weights = coordinator.decompress_model(compressed_weights)
        
        # Should be approximately equal (lossy compression)
        mse = torch.mean((model_weights - decompressed_weights) ** 2)
        assert mse < 0.1  # Reasonable compression error
    
    @pytest.mark.asyncio
    async def test_asynchronous_federated_learning(self, coordinator, mock_clients):
        """Test asynchronous federated learning with client updates."""
        # Register clients
        for client in mock_clients[:5]:
            await coordinator.register_client(client)
        
        # Enable asynchronous mode
        coordinator.set_async_mode(True)
        
        # Simulate asynchronous client updates
        async def simulate_client_update(client_id, delay):
            await asyncio.sleep(delay)
            update = {
                "client_id": client_id,
                "model_weights": torch.randn(100),
                "num_samples": 1000,
                "timestamp": datetime.now()
            }
            return await coordinator.process_async_update(update)
        
        # Start multiple client updates with different delays
        tasks = []
        for i, client in enumerate(mock_clients[:3]):
            task = asyncio.create_task(
                simulate_client_update(client.client_id, i * 0.1)
            )
            tasks.append(task)
        
        # Wait for all updates
        results = await asyncio.gather(*tasks)
        
        assert all(result.success for result in results)
        assert coordinator.get_pending_updates() >= 0
    
    def test_federated_evaluation_metrics(self, coordinator, mock_clients):
        """Test federated model evaluation and metrics."""
        # Register clients
        for client in mock_clients[:4]:
            coordinator.register_client(client)
        
        # Mock evaluation results from clients
        client_evaluations = []
        for client in mock_clients[:4]:
            evaluation = {
                "client_id": client.client_id,
                "accuracy": 0.8 + (np.random.rand() * 0.15),
                "loss": 0.3 + (np.random.rand() * 0.2),
                "num_test_samples": 200 + (np.random.randint(0, 100))
            }
            client_evaluations.append(evaluation)
        
        # Aggregate evaluation metrics
        federated_metrics = coordinator.aggregate_evaluation_metrics(client_evaluations)
        
        assert "weighted_accuracy" in federated_metrics
        assert "weighted_loss" in federated_metrics
        assert "total_test_samples" in federated_metrics
        assert "client_count" in federated_metrics
        
        assert 0.0 <= federated_metrics["weighted_accuracy"] <= 1.0
        assert federated_metrics["client_count"] == 4
    
    @pytest.mark.asyncio
    async def test_knowledge_distillation(self, coordinator):
        """Test knowledge distillation for model compression."""
        # Create teacher and student models
        teacher_model = torch.nn.Linear(10, 5)
        student_model = torch.nn.Linear(10, 5)
        
        knowledge_distillation = coordinator.knowledge_distillation
        
        # Mock training data
        training_data = [(torch.randn(32, 10), torch.randint(0, 5, (32,))) for _ in range(10)]
        
        # Perform knowledge distillation
        with patch.object(knowledge_distillation, '_distill_knowledge') as mock_distill:
            mock_distill.return_value = {
                "student_loss": 0.3,
                "distillation_loss": 0.2,
                "accuracy": 0.85
            }
            
            distillation_result = await knowledge_distillation.distill(
                teacher_model=teacher_model,
                student_model=student_model,
                training_data=training_data,
                temperature=3.0,
                alpha=0.7
            )
            
            assert distillation_result["student_loss"] < 1.0
            assert distillation_result["accuracy"] > 0.5
            mock_distill.assert_called_once()


class TestFederatedClient:
    """Test suite for Federated Client functionality."""
    
    @pytest.fixture
    def federated_client(self):
        """Create FederatedClient instance for testing."""
        client_config = {
            "client_id": "test_client",
            "data_path": "/mock/data/path",
            "model_config": {"hidden_size": 128, "num_layers": 2},
            "privacy_budget": {"epsilon": 1.0, "delta": 1e-5},
            "local_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.01
        }
        return FederatedClient(config=client_config)
    
    def test_client_initialization(self, federated_client):
        """Test FederatedClient initialization."""
        assert federated_client.client_id == "test_client"
        assert federated_client.local_model is not None
        assert federated_client.privacy_engine is not None
        assert federated_client.data_loader is not None
        assert federated_client.status == ClientStatus.INACTIVE
    
    @pytest.mark.asyncio
    async def test_local_model_training(self, federated_client):
        """Test local model training on client."""
        # Mock training data
        with patch.object(federated_client, '_load_training_data') as mock_data:
            mock_data.return_value = [(torch.randn(32, 10), torch.randint(0, 2, (32,))) for _ in range(5)]
            
            # Train local model
            training_result = await federated_client.train_local_model(
                global_weights=torch.randn(100),
                epochs=3
            )
            
            assert training_result["success"]
            assert "model_update" in training_result
            assert "num_samples" in training_result
            assert "training_loss" in training_result
            assert training_result["num_samples"] > 0
    
    def test_model_update_computation(self, federated_client):
        """Test computation of model updates (gradients/weights)."""
        # Set initial model weights
        initial_weights = torch.randn(100)
        federated_client.set_model_weights(initial_weights)
        
        # Simulate training (modify weights)
        trained_weights = initial_weights + torch.randn(100) * 0.1
        federated_client.set_model_weights(trained_weights)
        
        # Compute model update
        model_update = federated_client.compute_model_update(initial_weights)
        
        assert model_update is not None
        assert model_update.shape == initial_weights.shape
        
        # Update should be the difference
        expected_update = trained_weights - initial_weights
        assert torch.allclose(model_update, expected_update, atol=1e-6)
    
    def test_privacy_preserving_updates(self, federated_client):
        """Test privacy-preserving model updates."""
        # Create model update
        model_update = torch.randn(100)
        
        # Apply differential privacy
        private_update = federated_client.apply_differential_privacy(
            model_update=model_update,
            epsilon=1.0,
            delta=1e-5
        )
        
        assert private_update.shape == model_update.shape
        assert not torch.equal(private_update, model_update)  # Should be different due to noise
        
        # Check privacy budget consumption
        remaining_budget = federated_client.get_remaining_privacy_budget()
        assert remaining_budget.epsilon < 1.0  # Should be reduced
    
    @pytest.mark.asyncio
    async def test_client_evaluation(self, federated_client):
        """Test client model evaluation."""
        # Mock test data
        with patch.object(federated_client, '_load_test_data') as mock_test_data:
            mock_test_data.return_value = [(torch.randn(32, 10), torch.randint(0, 2, (32,))) for _ in range(3)]
            
            # Evaluate model
            evaluation_result = await federated_client.evaluate_model()
            
            assert "accuracy" in evaluation_result
            assert "loss" in evaluation_result
            assert "num_test_samples" in evaluation_result
            
            assert 0.0 <= evaluation_result["accuracy"] <= 1.0
            assert evaluation_result["num_test_samples"] > 0
    
    def test_data_statistics_computation(self, federated_client):
        """Test computation of local data statistics."""
        # Mock local data
        with patch.object(federated_client, '_get_local_data') as mock_data:
            mock_data.return_value = {
                "num_samples": 1500,
                "num_features": 20,
                "class_distribution": {0: 800, 1: 700},
                "data_quality_score": 0.92
            }
            
            stats = federated_client.compute_data_statistics()
            
            assert stats["num_samples"] == 1500
            assert stats["num_features"] == 20
            assert len(stats["class_distribution"]) == 2
            assert 0.0 <= stats["data_quality_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_secure_communication(self, federated_client):
        """Test secure communication with coordinator."""
        # Mock message to send
        message = {
            "type": "model_update",
            "data": torch.randn(100),
            "metadata": {"epoch": 5, "loss": 0.3}
        }
        
        # Encrypt message
        encrypted_message = federated_client.encrypt_message(message)
        
        assert encrypted_message != message
        assert "encrypted_data" in encrypted_message
        assert "signature" in encrypted_message
        
        # Decrypt message
        decrypted_message = federated_client.decrypt_message(encrypted_message)
        
        assert decrypted_message["type"] == message["type"]
        assert torch.equal(decrypted_message["data"], message["data"])


class TestModelAggregator:
    """Test suite for Model Aggregator functionality."""
    
    @pytest.fixture
    def model_aggregator(self):
        """Create ModelAggregator instance for testing."""
        return ModelAggregator()
    
    def test_federated_averaging(self, model_aggregator):
        """Test FedAvg aggregation algorithm."""
        # Create mock client updates
        client_updates = []
        for i in range(4):
            update = {
                "client_id": f"client_{i}",
                "model_weights": torch.randn(50) + i,  # Different weights per client
                "num_samples": 1000 + (i * 200)
            }
            client_updates.append(update)
        
        # Aggregate using FedAvg
        aggregated_weights = model_aggregator.federated_averaging(client_updates)
        
        assert aggregated_weights.shape == (50,)
        
        # Verify weighted average computation
        total_samples = sum(update["num_samples"] for update in client_updates)
        expected_weights = torch.zeros(50)
        for update in client_updates:
            weight = update["num_samples"] / total_samples
            expected_weights += weight * update["model_weights"]
        
        assert torch.allclose(aggregated_weights, expected_weights, atol=1e-6)
    
    def test_federated_proximal_aggregation(self, model_aggregator):
        """Test FedProx aggregation algorithm."""
        # Create client updates with proximal term
        client_updates = []
        global_model = torch.randn(50)
        
        for i in range(3):
            update = {
                "client_id": f"client_{i}",
                "model_weights": torch.randn(50),
                "num_samples": 1000,
                "proximal_term": 0.1  # Regularization strength
            }
            client_updates.append(update)
        
        # Aggregate using FedProx
        aggregated_weights = model_aggregator.federated_proximal(
            client_updates=client_updates,
            global_model=global_model,
            mu=0.1
        )
        
        assert aggregated_weights.shape == (50,)
        assert not torch.equal(aggregated_weights, global_model)
    
    def test_adaptive_aggregation(self, model_aggregator):
        """Test adaptive aggregation based on client performance."""
        # Create client updates with performance metrics
        client_updates = []
        for i in range(4):
            update = {
                "client_id": f"client_{i}",
                "model_weights": torch.randn(50),
                "num_samples": 1000,
                "accuracy": 0.7 + (i * 0.05),  # Varying performance
                "loss": 0.5 - (i * 0.05)
            }
            client_updates.append(update)
        
        # Aggregate with adaptive weighting
        aggregated_weights = model_aggregator.adaptive_aggregation(
            client_updates=client_updates,
            weighting_strategy="performance"
        )
        
        assert aggregated_weights.shape == (50,)
        
        # Higher performing clients should have more influence
        # This is tested implicitly through the aggregation process
    
    def test_robust_aggregation(self, model_aggregator):
        """Test robust aggregation against Byzantine clients."""
        # Create client updates with some outliers
        client_updates = []
        for i in range(6):
            if i < 4:  # Normal clients
                weights = torch.randn(50)
            else:  # Byzantine clients with extreme values
                weights = torch.randn(50) * 10
            
            update = {
                "client_id": f"client_{i}",
                "model_weights": weights,
                "num_samples": 1000
            }
            client_updates.append(update)
        
        # Aggregate with Byzantine robustness
        aggregated_weights = model_aggregator.robust_aggregation(
            client_updates=client_updates,
            byzantine_tolerance=2  # Can tolerate 2 Byzantine clients
        )
        
        assert aggregated_weights.shape == (50,)
        
        # Result should not be dominated by outliers
        assert torch.norm(aggregated_weights) < 5.0  # Reasonable magnitude
    
    def test_compression_aware_aggregation(self, model_aggregator):
        """Test aggregation with compressed model updates."""
        # Create compressed client updates
        client_updates = []
        for i in range(3):
            # Simulate compression by quantizing weights
            original_weights = torch.randn(50)
            compressed_weights = torch.round(original_weights * 8) / 8  # 3-bit quantization
            
            update = {
                "client_id": f"client_{i}",
                "model_weights": compressed_weights,
                "num_samples": 1000,
                "compression_ratio": 0.375  # 3 bits / 8 bits
            }
            client_updates.append(update)
        
        # Aggregate compressed updates
        aggregated_weights = model_aggregator.compression_aware_aggregation(client_updates)
        
        assert aggregated_weights.shape == (50,)
        # Aggregation should handle quantization artifacts


class TestPrivacyEngine:
    """Test suite for Privacy Engine functionality."""
    
    @pytest.fixture
    def privacy_engine(self):
        """Create PrivacyEngine instance for testing."""
        config = {
            "differential_privacy_enabled": True,
            "secure_aggregation_enabled": True,
            "homomorphic_encryption": False,
            "privacy_budget_epsilon": 1.0,
            "privacy_budget_delta": 1e-5
        }
        return PrivacyEngine(config=config)
    
    def test_differential_privacy_noise_calibration(self, privacy_engine):
        """Test differential privacy noise calibration."""
        dp = privacy_engine.differential_privacy
        
        # Test Gaussian noise calibration
        epsilon = 1.0
        delta = 1e-5
        sensitivity = 1.0
        
        noise_scale = dp.calibrate_gaussian_noise(epsilon, delta, sensitivity)
        
        assert noise_scale > 0
        assert isinstance(noise_scale, float)
        
        # Test Laplace noise calibration
        laplace_scale = dp.calibrate_laplace_noise(epsilon, sensitivity)
        
        assert laplace_scale > 0
        assert isinstance(laplace_scale, float)
    
    def test_privacy_accounting(self, privacy_engine):
        """Test privacy budget accounting and composition."""
        dp = privacy_engine.differential_privacy
        
        # Initialize privacy accountant
        accountant = dp.create_privacy_accountant(
            initial_epsilon=2.0,
            initial_delta=1e-4
        )
        
        # Consume privacy budget
        consumption1 = PrivacyBudget(epsilon=0.5, delta=1e-5)
        remaining1 = accountant.consume_budget(consumption1)
        
        assert remaining1.epsilon == 1.5
        assert remaining1.delta == 9e-5
        
        # Consume more budget
        consumption2 = PrivacyBudget(epsilon=0.8, delta=2e-5)
        remaining2 = accountant.consume_budget(consumption2)
        
        assert remaining2.epsilon == 0.7
        assert remaining2.delta == 7e-5
        
        # Check if budget is exhausted
        large_consumption = PrivacyBudget(epsilon=1.0, delta=1e-4)
        with pytest.raises(ValueError, match="Insufficient privacy budget"):
            accountant.consume_budget(large_consumption)
    
    def test_secure_multiparty_computation(self, privacy_engine):
        """Test secure multi-party computation protocols."""
        smc = privacy_engine.secure_aggregation
        
        # Create secret shares for multiple parties
        secret_value = torch.tensor([1.5, 2.3, 3.7, 4.1])
        num_parties = 4
        threshold = 3
        
        shares = smc.create_secret_shares(
            secret=secret_value,
            num_parties=num_parties,
            threshold=threshold
        )
        
        assert len(shares) == num_parties
        
        # Reconstruct secret from threshold shares
        reconstructed = smc.reconstruct_secret(shares[:threshold])
        
        assert torch.allclose(reconstructed, secret_value, atol=1e-6)
        
        # Should fail with insufficient shares
        with pytest.raises(ValueError, match="Insufficient shares"):
            smc.reconstruct_secret(shares[:threshold-1])
    
    def test_homomorphic_encryption_operations(self, privacy_engine):
        """Test homomorphic encryption for secure computation."""
        # Enable homomorphic encryption
        privacy_engine.config["homomorphic_encryption"] = True
        he = privacy_engine.homomorphic_encryption
        
        # Generate keys
        public_key, private_key = he.generate_keys()
        
        # Encrypt values
        value1 = 5.0
        value2 = 3.0
        
        encrypted1 = he.encrypt(value1, public_key)
        encrypted2 = he.encrypt(value2, public_key)
        
        # Perform homomorphic addition
        encrypted_sum = he.add(encrypted1, encrypted2)
        
        # Decrypt result
        decrypted_sum = he.decrypt(encrypted_sum, private_key)
        
        assert abs(decrypted_sum - (value1 + value2)) < 1e-6
        
        # Perform homomorphic multiplication
        encrypted_product = he.multiply(encrypted1, encrypted2)
        decrypted_product = he.decrypt(encrypted_product, private_key)
        
        assert abs(decrypted_product - (value1 * value2)) < 1e-6
    
    def test_privacy_preserving_metrics(self, privacy_engine):
        """Test privacy-preserving computation of metrics."""
        # Mock client metrics with noise
        client_metrics = [
            {"accuracy": 0.85, "num_samples": 1000},
            {"accuracy": 0.82, "num_samples": 1200},
            {"accuracy": 0.88, "num_samples": 800}
        ]
        
        # Compute privacy-preserving average
        private_average = privacy_engine.compute_private_average(
            values=[m["accuracy"] for m in client_metrics],
            weights=[m["num_samples"] for m in client_metrics],
            epsilon=1.0,
            delta=1e-5
        )
        
        # Should be close to true weighted average but with noise
        true_average = sum(m["accuracy"] * m["num_samples"] for m in client_metrics) / sum(m["num_samples"] for m in client_metrics)
        
        assert abs(private_average - true_average) < 0.1  # Reasonable noise level
    
    def test_membership_inference_protection(self, privacy_engine):
        """Test protection against membership inference attacks."""
        # Create model outputs for members and non-members
        member_outputs = torch.randn(100, 10)  # 100 samples, 10 classes
        non_member_outputs = torch.randn(100, 10)
        
        # Apply membership inference protection
        protected_outputs = privacy_engine.protect_against_membership_inference(
            model_outputs=torch.cat([member_outputs, non_member_outputs]),
            epsilon=1.0,
            protection_method="output_perturbation"
        )
        
        assert protected_outputs.shape == (200, 10)
        assert not torch.equal(protected_outputs, torch.cat([member_outputs, non_member_outputs]))
        
        # Test membership inference attack resistance
        attack_success_rate = privacy_engine.evaluate_membership_inference_resistance(
            protected_outputs=protected_outputs,
            member_indices=list(range(100)),
            non_member_indices=list(range(100, 200))
        )
        
        # Attack success rate should be close to random guessing (0.5)
        assert 0.4 <= attack_success_rate <= 0.6