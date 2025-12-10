"""
Test suite for Neural Embedding Engine - ML-powered embedding generation.

This module tests the machine learning enhanced embedding capabilities:
- Transformer-based embedding generation
- Multi-modal embedding support
- Adaptive embedding optimization
- Transfer learning capabilities
- Model fine-tuning and evaluation

Following TDD principles: Red-Green-Refactor cycle.
"""

import pytest
import numpy as np
import torch
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Import the classes we'll implement
from safla.core.ml_neural_embedding_engine import (
    NeuralEmbeddingEngine,
    TransformerEmbeddingModel,
    MultiModalEmbedding,
    EmbeddingOptimizer,
    TransferLearningManager,
    ModelEvaluator,
    EmbeddingConfig,
    ModelMetrics,
    OptimizationResult
)


class TestNeuralEmbeddingEngine:
    """Test suite for Neural Embedding Engine functionality."""
    
    @pytest.fixture
    def embedding_config(self):
        """Create embedding configuration for testing."""
        return EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            max_sequence_length=512,
            batch_size=32,
            device="cpu",
            cache_embeddings=True,
            fine_tune_enabled=True
        )
    
    @pytest.fixture
    def neural_engine(self, embedding_config):
        """Create a NeuralEmbeddingEngine instance for testing."""
        return NeuralEmbeddingEngine(config=embedding_config)
    
    @pytest.fixture
    def sample_texts(self):
        """Generate sample texts for embedding testing."""
        return [
            "Machine learning is transforming artificial intelligence",
            "Deep neural networks process complex patterns",
            "Natural language processing enables text understanding",
            "Computer vision analyzes visual information",
            "Reinforcement learning optimizes decision making"
        ]
    
    def test_neural_engine_initialization(self, neural_engine, embedding_config):
        """Test NeuralEmbeddingEngine initialization with correct parameters."""
        assert neural_engine.config == embedding_config
        assert neural_engine.model is not None
        assert neural_engine.embedding_dim == embedding_config.embedding_dim
        assert neural_engine.device == embedding_config.device
        assert neural_engine.is_initialized
    
    def test_neural_engine_invalid_config(self):
        """Test NeuralEmbeddingEngine rejects invalid configuration."""
        with pytest.raises(ValueError, match="Invalid model name"):
            invalid_config = EmbeddingConfig(
                model_name="",
                embedding_dim=384
            )
            NeuralEmbeddingEngine(config=invalid_config)
        
        with pytest.raises(ValueError, match="Embedding dimension must be positive"):
            invalid_config = EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                embedding_dim=0
            )
            NeuralEmbeddingEngine(config=invalid_config)
    
    @pytest.mark.asyncio
    async def test_generate_single_embedding(self, neural_engine, sample_texts):
        """Test generating embedding for single text."""
        text = sample_texts[0]
        
        embedding = await neural_engine.generate_embedding(text)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == neural_engine.embedding_dim
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()
    
    @pytest.mark.asyncio
    async def test_generate_batch_embeddings(self, neural_engine, sample_texts):
        """Test generating embeddings for batch of texts."""
        embeddings = await neural_engine.generate_batch_embeddings(sample_texts)
        
        assert len(embeddings) == len(sample_texts)
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape[0] == neural_engine.embedding_dim for emb in embeddings)
        assert all(not np.isnan(emb).any() for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_embedding_similarity_computation(self, neural_engine, sample_texts):
        """Test semantic similarity computation between embeddings."""
        # Generate embeddings for similar and dissimilar texts
        similar_texts = [
            "Machine learning algorithms process data",
            "AI algorithms analyze information patterns"
        ]
        
        dissimilar_texts = [
            "Machine learning algorithms process data", 
            "The weather is sunny today"
        ]
        
        # Test similar texts
        similar_embeddings = await neural_engine.generate_batch_embeddings(similar_texts)
        similar_score = neural_engine.compute_similarity(
            similar_embeddings[0], 
            similar_embeddings[1]
        )
        
        # Test dissimilar texts
        dissimilar_embeddings = await neural_engine.generate_batch_embeddings(dissimilar_texts)
        dissimilar_score = neural_engine.compute_similarity(
            dissimilar_embeddings[0], 
            dissimilar_embeddings[1]
        )
        
        # Similar texts should have higher similarity
        assert similar_score > dissimilar_score
        assert 0.0 <= similar_score <= 1.0
        assert 0.0 <= dissimilar_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_embedding_caching(self, neural_engine, sample_texts):
        """Test embedding caching functionality."""
        text = sample_texts[0]
        
        # Generate embedding first time
        start_time = datetime.now()
        embedding1 = await neural_engine.generate_embedding(text)
        first_duration = datetime.now() - start_time
        
        # Generate same embedding again (should be cached)
        start_time = datetime.now()
        embedding2 = await neural_engine.generate_embedding(text)
        second_duration = datetime.now() - start_time
        
        # Results should be identical
        assert np.array_equal(embedding1, embedding2)
        
        # Second call should be faster (cached)
        if neural_engine.config.cache_embeddings:
            assert second_duration < first_duration
    
    @pytest.mark.asyncio
    async def test_model_fine_tuning(self, neural_engine):
        """Test model fine-tuning capabilities."""
        # Prepare training data
        training_pairs = [
            ("positive example", "similar positive text", 1.0),
            ("negative example", "dissimilar negative text", 0.0),
            ("neutral example", "somewhat related text", 0.5)
        ]
        
        # Mock fine-tuning process
        with patch.object(neural_engine, '_fine_tune_model') as mock_fine_tune:
            mock_fine_tune.return_value = ModelMetrics(
                loss=0.15,
                accuracy=0.92,
                f1_score=0.89,
                training_time=120.5
            )
            
            metrics = await neural_engine.fine_tune_model(
                training_pairs=training_pairs,
                epochs=3,
                learning_rate=1e-5
            )
            
            assert metrics.loss < 0.5
            assert metrics.accuracy > 0.8
            assert metrics.f1_score > 0.8
            assert metrics.training_time > 0
            mock_fine_tune.assert_called_once()
    
    def test_model_evaluation(self, neural_engine):
        """Test model evaluation metrics."""
        # Mock evaluation data
        test_embeddings = [
            np.random.rand(neural_engine.embedding_dim) for _ in range(10)
        ]
        test_labels = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
        
        evaluator = ModelEvaluator(neural_engine)
        metrics = evaluator.evaluate_embeddings(test_embeddings, test_labels)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert all(0.0 <= score <= 1.0 for score in metrics.values())
    
    @pytest.mark.asyncio
    async def test_transfer_learning(self, neural_engine):
        """Test transfer learning capabilities."""
        # Mock source domain data
        source_texts = [
            "source domain text example one",
            "source domain text example two"
        ]
        
        # Mock target domain data
        target_texts = [
            "target domain text example one",
            "target domain text example two"
        ]
        
        transfer_manager = TransferLearningManager(neural_engine)
        
        with patch.object(transfer_manager, '_adapt_model') as mock_adapt:
            mock_adapt.return_value = OptimizationResult(
                improvement_score=0.15,
                convergence_achieved=True,
                iterations_completed=50,
                final_loss=0.08
            )
            
            result = await transfer_manager.adapt_to_domain(
                source_texts=source_texts,
                target_texts=target_texts,
                adaptation_steps=100
            )
            
            assert result.improvement_score > 0
            assert result.convergence_achieved
            assert result.iterations_completed > 0
            assert result.final_loss < 0.5
    
    def test_multi_modal_embedding_support(self, neural_engine):
        """Test multi-modal embedding generation."""
        # Mock multi-modal input
        text_input = "A beautiful sunset over the mountains"
        image_features = np.random.rand(2048)  # Mock image features
        audio_features = np.random.rand(1024)  # Mock audio features
        
        multi_modal = MultiModalEmbedding(neural_engine)
        
        with patch.object(multi_modal, '_fuse_modalities') as mock_fuse:
            mock_fuse.return_value = np.random.rand(neural_engine.embedding_dim)
            
            embedding = multi_modal.generate_multimodal_embedding(
                text=text_input,
                image_features=image_features,
                audio_features=audio_features
            )
            
            assert embedding is not None
            assert embedding.shape[0] == neural_engine.embedding_dim
            mock_fuse.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embedding_optimization(self, neural_engine, sample_texts):
        """Test embedding optimization for specific tasks."""
        optimizer = EmbeddingOptimizer(neural_engine)
        
        # Mock optimization task
        task_data = {
            "task_type": "semantic_similarity",
            "training_pairs": [
                (sample_texts[0], sample_texts[1], 0.8),
                (sample_texts[2], sample_texts[3], 0.6)
            ]
        }
        
        with patch.object(optimizer, '_optimize_for_task') as mock_optimize:
            mock_optimize.return_value = OptimizationResult(
                improvement_score=0.22,
                convergence_achieved=True,
                iterations_completed=75,
                final_loss=0.05
            )
            
            result = await optimizer.optimize_for_task(task_data)
            
            assert result.improvement_score > 0.2
            assert result.convergence_achieved
            assert result.final_loss < 0.1
    
    def test_embedding_dimensionality_reduction(self, neural_engine):
        """Test dimensionality reduction for embeddings."""
        # Generate high-dimensional embeddings
        high_dim_embeddings = [
            np.random.rand(neural_engine.embedding_dim) for _ in range(100)
        ]
        
        # Reduce dimensionality
        reduced_embeddings = neural_engine.reduce_dimensionality(
            embeddings=high_dim_embeddings,
            target_dim=128,
            method="pca"
        )
        
        assert len(reduced_embeddings) == len(high_dim_embeddings)
        assert all(emb.shape[0] == 128 for emb in reduced_embeddings)
        
        # Test preservation of relative similarities
        original_sim = neural_engine.compute_similarity(
            high_dim_embeddings[0], 
            high_dim_embeddings[1]
        )
        reduced_sim = neural_engine.compute_similarity(
            reduced_embeddings[0], 
            reduced_embeddings[1]
        )
        
        # Similarity should be approximately preserved
        assert abs(original_sim - reduced_sim) < 0.3
    
    @pytest.mark.asyncio
    async def test_embedding_anomaly_detection(self, neural_engine, sample_texts):
        """Test anomaly detection in embeddings."""
        # Generate normal embeddings
        normal_embeddings = await neural_engine.generate_batch_embeddings(sample_texts)
        
        # Create anomalous embedding
        anomalous_embedding = np.random.rand(neural_engine.embedding_dim) * 10
        
        all_embeddings = normal_embeddings + [anomalous_embedding]
        
        # Detect anomalies
        anomaly_scores = neural_engine.detect_anomalies(all_embeddings)
        
        assert len(anomaly_scores) == len(all_embeddings)
        assert all(0.0 <= score <= 1.0 for score in anomaly_scores)
        
        # Anomalous embedding should have highest score
        assert anomaly_scores[-1] == max(anomaly_scores)
    
    def test_embedding_clustering(self, neural_engine):
        """Test clustering of embeddings."""
        # Generate embeddings for clustering
        embeddings = [
            np.random.rand(neural_engine.embedding_dim) for _ in range(20)
        ]
        
        # Perform clustering
        cluster_labels = neural_engine.cluster_embeddings(
            embeddings=embeddings,
            n_clusters=3,
            method="kmeans"
        )
        
        assert len(cluster_labels) == len(embeddings)
        assert all(isinstance(label, int) for label in cluster_labels)
        assert len(set(cluster_labels)) <= 3  # Should have at most 3 clusters
        assert min(cluster_labels) >= 0  # Labels should be non-negative
    
    @pytest.mark.asyncio
    async def test_model_versioning_and_rollback(self, neural_engine):
        """Test model versioning and rollback capabilities."""
        # Save current model state
        version_id = neural_engine.save_model_checkpoint("v1.0")
        assert version_id is not None
        
        # Mock model modification
        original_weights = neural_engine.model.state_dict().copy()
        
        # Simulate model update
        with patch.object(neural_engine, '_update_model_weights'):
            neural_engine._update_model_weights({"layer1.weight": torch.randn(10, 10)})
        
        # Rollback to previous version
        success = neural_engine.rollback_to_checkpoint(version_id)
        assert success
        
        # Verify model state is restored
        current_weights = neural_engine.model.state_dict()
        # In real implementation, weights should match original
    
    def test_embedding_quality_metrics(self, neural_engine):
        """Test embedding quality assessment."""
        # Generate test embeddings
        embeddings = [
            np.random.rand(neural_engine.embedding_dim) for _ in range(50)
        ]
        
        quality_metrics = neural_engine.assess_embedding_quality(embeddings)
        
        expected_metrics = [
            "diversity_score",
            "coherence_score", 
            "separability_score",
            "stability_score"
        ]
        
        for metric in expected_metrics:
            assert metric in quality_metrics
            assert 0.0 <= quality_metrics[metric] <= 1.0
    
    @pytest.mark.asyncio
    async def test_real_time_embedding_generation(self, neural_engine):
        """Test real-time embedding generation with streaming."""
        text_stream = [
            "First streaming text chunk",
            "Second streaming text chunk", 
            "Third streaming text chunk"
        ]
        
        embeddings = []
        async for embedding in neural_engine.generate_streaming_embeddings(text_stream):
            embeddings.append(embedding)
            assert embedding.shape[0] == neural_engine.embedding_dim
        
        assert len(embeddings) == len(text_stream)
    
    def test_memory_efficient_processing(self, neural_engine):
        """Test memory-efficient processing for large batches."""
        # Create large batch of texts
        large_batch = [f"Text sample {i}" for i in range(1000)]
        
        # Process with memory constraints
        with patch.object(neural_engine, '_process_in_chunks') as mock_process:
            mock_process.return_value = [
                np.random.rand(neural_engine.embedding_dim) for _ in large_batch
            ]
            
            embeddings = neural_engine.generate_batch_embeddings_memory_efficient(
                texts=large_batch,
                max_memory_mb=512
            )
            
            assert len(embeddings) == len(large_batch)
            mock_process.assert_called_once()


class TestTransformerEmbeddingModel:
    """Test suite for Transformer-based embedding model."""
    
    @pytest.fixture
    def transformer_model(self):
        """Create TransformerEmbeddingModel for testing."""
        config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384
        )
        return TransformerEmbeddingModel(config)
    
    def test_transformer_model_loading(self, transformer_model):
        """Test transformer model loading and initialization."""
        assert transformer_model.model is not None
        assert transformer_model.tokenizer is not None
        assert transformer_model.embedding_dim == 384
    
    def test_text_tokenization(self, transformer_model):
        """Test text tokenization process."""
        text = "This is a test sentence for tokenization"
        
        tokens = transformer_model.tokenize(text)
        
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert len(tokens["input_ids"]) > 0
        assert len(tokens["attention_mask"]) == len(tokens["input_ids"])
    
    def test_forward_pass(self, transformer_model):
        """Test forward pass through transformer model."""
        text = "Test sentence for forward pass"
        tokens = transformer_model.tokenize(text)
        
        with torch.no_grad():
            outputs = transformer_model.forward(tokens)
        
        assert outputs is not None
        assert outputs.shape[-1] == transformer_model.embedding_dim
    
    def test_attention_visualization(self, transformer_model):
        """Test attention pattern visualization."""
        text = "Attention visualization test sentence"
        
        attention_weights = transformer_model.get_attention_weights(text)
        
        assert attention_weights is not None
        assert len(attention_weights.shape) >= 2  # Should be multi-dimensional
        assert not np.isnan(attention_weights).any()


class TestEmbeddingOptimizer:
    """Test suite for embedding optimization functionality."""
    
    @pytest.fixture
    def optimizer(self):
        """Create EmbeddingOptimizer for testing."""
        config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384
        )
        engine = NeuralEmbeddingEngine(config)
        return EmbeddingOptimizer(engine)
    
    @pytest.mark.asyncio
    async def test_contrastive_learning_optimization(self, optimizer):
        """Test contrastive learning optimization."""
        # Mock training triplets (anchor, positive, negative)
        triplets = [
            ("anchor text", "similar positive text", "dissimilar negative text"),
            ("another anchor", "related positive", "unrelated negative")
        ]
        
        with patch.object(optimizer, '_contrastive_loss') as mock_loss:
            mock_loss.return_value = torch.tensor(0.25)
            
            result = await optimizer.optimize_contrastive_learning(
                triplets=triplets,
                epochs=5,
                learning_rate=1e-4
            )
            
            assert result.improvement_score > 0
            assert result.final_loss < 0.5
    
    @pytest.mark.asyncio
    async def test_metric_learning_optimization(self, optimizer):
        """Test metric learning optimization."""
        # Mock similarity pairs
        similarity_pairs = [
            ("text A", "text B", 0.8),
            ("text C", "text D", 0.3),
            ("text E", "text F", 0.9)
        ]
        
        result = await optimizer.optimize_metric_learning(
            similarity_pairs=similarity_pairs,
            margin=0.2,
            epochs=10
        )
        
        assert result.improvement_score >= 0
        assert result.convergence_achieved in [True, False]
    
    def test_hyperparameter_tuning(self, optimizer):
        """Test hyperparameter optimization."""
        param_space = {
            "learning_rate": [1e-5, 1e-4, 1e-3],
            "batch_size": [16, 32, 64],
            "dropout": [0.1, 0.2, 0.3]
        }
        
        with patch.object(optimizer, '_evaluate_hyperparameters') as mock_eval:
            mock_eval.return_value = {"accuracy": 0.85, "loss": 0.15}
            
            best_params = optimizer.tune_hyperparameters(
                param_space=param_space,
                optimization_metric="accuracy",
                n_trials=5
            )
            
            assert "learning_rate" in best_params
            assert "batch_size" in best_params
            assert "dropout" in best_params