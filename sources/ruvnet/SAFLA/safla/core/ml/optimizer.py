"""
Embedding optimization module for the SAFLA ML system.

This module provides optimization strategies and techniques for improving
embedding quality and model performance.
"""

import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from .base import EmbeddingConfig, OptimizationResult, ModelMetrics, EvaluationMetrics
from .models import TransformerEmbeddingModel, ContrastiveLoss, TripletLoss

logger = logging.getLogger(__name__)


class EmbeddingOptimizer:
    """Optimizer for embedding quality and performance."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.optimization_history = []
        self.best_metrics = None
    
    def optimize_embeddings(self, model: TransformerEmbeddingModel, 
                          training_data: List[Dict[str, Any]]) -> OptimizationResult:
        """Optimize embedding model using specified strategy."""
        start_time = datetime.now()
        
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=self.config.warmup_steps
        )
        
        # Initialize result tracking
        optimization_result = OptimizationResult(
            strategy_used=self.config.optimization_strategy.value
        )
        
        # Initial evaluation
        initial_metrics = self._evaluate_embeddings(model, training_data[:100])
        logger.info(f"Initial metrics: {initial_metrics.embedding_quality_score:.4f}")
        
        # Training loop
        model.train()
        total_loss = 0.0
        iterations = 0
        
        try:
            for epoch in range(10):  # Limited epochs for optimization
                epoch_loss = 0.0
                epoch_batches = 0
                
                for batch_idx, batch in enumerate(self._create_batches(training_data)):
                    optimizer.zero_grad()
                    
                    # Compute loss based on optimization strategy
                    loss = self._compute_optimization_loss(model, batch)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Invalid loss at epoch {epoch}, batch {batch_idx}")
                        continue
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    total_loss += loss.item()
                    epochs_batches += 1
                    iterations += 1
                    
                    # Periodic evaluation
                    if batch_idx % self.config.evaluation_frequency == 0:
                        eval_metrics = self._evaluate_embeddings(model, training_data[:50])
                        logger.debug(f"Epoch {epoch}, Batch {batch_idx}: "
                                   f"Quality={eval_metrics.embedding_quality_score:.4f}")
                
                avg_epoch_loss = epoch_loss / max(1, epoch_batches)
                logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
        
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            optimization_result.success = False
            return optimization_result
        
        # Final evaluation
        model.eval()
        final_metrics = self._evaluate_embeddings(model, training_data[:100])
        
        # Calculate improvement
        improvement = (final_metrics.embedding_quality_score - 
                      initial_metrics.embedding_quality_score)
        
        # Prepare result
        optimization_result.success = True
        optimization_result.improvement = improvement
        optimization_result.iterations = iterations
        optimization_result.convergence_time = (datetime.now() - start_time).total_seconds()
        optimization_result.metrics = ModelMetrics(
            training_loss=total_loss / max(1, iterations),
            embedding_quality=final_metrics.embedding_quality_score,
            inference_time=final_metrics.silhouette_score,  # Temporary mapping
            timestamp=datetime.now()
        )
        
        # Store in history
        self.optimization_history.append({
            "timestamp": datetime.now(),
            "strategy": self.config.optimization_strategy.value,
            "improvement": improvement,
            "final_quality": final_metrics.embedding_quality_score,
            "iterations": iterations
        })
        
        # Update best metrics if this is better
        if (self.best_metrics is None or 
            final_metrics.embedding_quality_score > self.best_metrics.embedding_quality_score):
            self.best_metrics = final_metrics
        
        logger.info(f"Optimization completed. Improvement: {improvement:.4f}")
        return optimization_result
    
    def _create_batches(self, training_data: List[Dict[str, Any]]):
        """Create training batches from data."""
        batch_size = self.config.batch_size
        
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]
            yield batch
    
    def _compute_optimization_loss(self, model: TransformerEmbeddingModel, 
                                 batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute loss based on optimization strategy."""
        if self.config.optimization_strategy.value == "contrastive":
            return self._compute_contrastive_loss(model, batch)
        elif self.config.optimization_strategy.value == "triplet":
            return self._compute_triplet_loss(model, batch)
        elif self.config.optimization_strategy.value == "supervised":
            return self._compute_supervised_loss(model, batch)
        else:
            # Default to contrastive
            return self._compute_contrastive_loss(model, batch)
    
    def _compute_contrastive_loss(self, model: TransformerEmbeddingModel, 
                                batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute contrastive loss for the batch."""
        texts = [item.get("text", str(item.get("content", ""))) for item in batch]
        
        if not texts or len(texts) < 2:
            return torch.tensor(0.0, requires_grad=True)
        
        # Tokenize inputs
        tokenizer = model.tokenizer
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt"
        ).to(model.device)
        
        # Get embeddings
        outputs = model(inputs)
        embeddings = outputs["embeddings"]
        
        # Apply contrastive head
        contrastive_features = model.get_contrastive_features(embeddings)
        
        # Compute contrastive loss
        contrastive_loss = ContrastiveLoss(temperature=0.1)
        loss = contrastive_loss(contrastive_features)
        
        return loss
    
    def _compute_triplet_loss(self, model: TransformerEmbeddingModel, 
                            batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute triplet loss for the batch."""
        # Extract triplets from batch
        anchors, positives, negatives = [], [], []
        
        for item in batch:
            if all(key in item for key in ["anchor", "positive", "negative"]):
                anchors.append(item["anchor"])
                positives.append(item["positive"])
                negatives.append(item["negative"])
        
        if not anchors:
            return torch.tensor(0.0, requires_grad=True)
        
        # Get embeddings for each set
        anchor_embeds = self._get_text_embeddings(model, anchors)
        positive_embeds = self._get_text_embeddings(model, positives)
        negative_embeds = self._get_text_embeddings(model, negatives)
        
        # Compute triplet loss
        triplet_loss = TripletLoss(margin=1.0)
        loss = triplet_loss(anchor_embeds, positive_embeds, negative_embeds)
        
        return loss
    
    def _compute_supervised_loss(self, model: TransformerEmbeddingModel, 
                               batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute supervised loss for the batch."""
        texts = []
        labels = []
        
        for item in batch:
            if "text" in item and "label" in item:
                texts.append(item["text"])
                labels.append(item["label"])
        
        if not texts:
            return torch.tensor(0.0, requires_grad=True)
        
        # Get embeddings
        embeddings = self._get_text_embeddings(model, texts)
        
        # Get classification logits
        logits = model.get_classification_logits(embeddings)
        
        # Convert labels to tensor
        label_tensor = torch.tensor(labels, dtype=torch.long, device=model.device)
        
        # Compute cross-entropy loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, label_tensor)
        
        return loss
    
    def _get_text_embeddings(self, model: TransformerEmbeddingModel, 
                           texts: List[str]) -> torch.Tensor:
        """Get embeddings for list of texts."""
        tokenizer = model.tokenizer
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model(inputs)
        return outputs["embeddings"]
    
    def _evaluate_embeddings(self, model: TransformerEmbeddingModel, 
                           test_data: List[Dict[str, Any]]) -> EvaluationMetrics:
        """Evaluate embedding quality using various metrics."""
        if not test_data:
            return EvaluationMetrics(
                silhouette_score=0.0,
                intra_cluster_distance=0.0,
                inter_cluster_distance=0.0,
                embedding_quality_score=0.0,
                retrieval_accuracy=0.0,
                clustering_purity=0.0,
                anomaly_detection_f1=0.0
            )
        
        model.eval()
        
        # Extract texts and labels if available
        texts = [item.get("text", str(item.get("content", ""))) for item in test_data]
        labels = [item.get("label", 0) for item in test_data if "label" in item]
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self._get_text_embeddings(model, texts)
            embeddings_np = embeddings.cpu().numpy()
        
        # Compute metrics
        try:
            # Silhouette score (requires at least 2 clusters)
            if len(set(labels)) >= 2 and len(labels) == len(texts):
                sil_score = silhouette_score(embeddings_np, labels)
            else:
                # Use K-means clustering
                if len(embeddings_np) >= 2:
                    kmeans = KMeans(n_clusters=min(3, len(embeddings_np)), random_state=42)
                    cluster_labels = kmeans.fit_predict(embeddings_np)
                    sil_score = silhouette_score(embeddings_np, cluster_labels)
                else:
                    sil_score = 0.0
            
            # Distance metrics
            intra_cluster_dist = self._compute_intra_cluster_distance(embeddings_np, labels)
            inter_cluster_dist = self._compute_inter_cluster_distance(embeddings_np, labels)
            
            # Overall quality score (combination of metrics)
            quality_score = (
                sil_score * 0.4 +
                (1.0 / (1.0 + intra_cluster_dist)) * 0.3 +
                inter_cluster_dist * 0.3
            )
            
            return EvaluationMetrics(
                silhouette_score=sil_score,
                intra_cluster_distance=intra_cluster_dist,
                inter_cluster_distance=inter_cluster_dist,
                embedding_quality_score=max(0.0, min(1.0, quality_score)),
                retrieval_accuracy=0.8,  # Placeholder
                clustering_purity=0.7,   # Placeholder
                anomaly_detection_f1=0.6  # Placeholder
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return EvaluationMetrics(
                silhouette_score=0.0,
                intra_cluster_distance=1.0,
                inter_cluster_distance=0.0,
                embedding_quality_score=0.0,
                retrieval_accuracy=0.0,
                clustering_purity=0.0,
                anomaly_detection_f1=0.0
            )
    
    def _compute_intra_cluster_distance(self, embeddings: np.ndarray, 
                                      labels: List[int]) -> float:
        """Compute average intra-cluster distance."""
        if not labels or len(set(labels)) < 2:
            return np.mean(np.linalg.norm(embeddings - embeddings.mean(axis=0), axis=1))
        
        total_dist = 0.0
        total_pairs = 0
        
        for label in set(labels):
            cluster_embeds = embeddings[np.array(labels) == label]
            if len(cluster_embeds) > 1:
                for i in range(len(cluster_embeds)):
                    for j in range(i + 1, len(cluster_embeds)):
                        total_dist += np.linalg.norm(cluster_embeds[i] - cluster_embeds[j])
                        total_pairs += 1
        
        return total_dist / max(1, total_pairs)
    
    def _compute_inter_cluster_distance(self, embeddings: np.ndarray, 
                                      labels: List[int]) -> float:
        """Compute average inter-cluster distance."""
        if not labels or len(set(labels)) < 2:
            return 0.0
        
        unique_labels = list(set(labels))
        total_dist = 0.0
        total_pairs = 0
        
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                embeds_i = embeddings[np.array(labels) == unique_labels[i]]
                embeds_j = embeddings[np.array(labels) == unique_labels[j]]
                
                center_i = embeds_i.mean(axis=0)
                center_j = embeds_j.mean(axis=0)
                
                total_dist += np.linalg.norm(center_i - center_j)
                total_pairs += 1
        
        return total_dist / max(1, total_pairs)
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of optimization runs."""
        return self.optimization_history.copy()
    
    def get_best_metrics(self) -> Optional[EvaluationMetrics]:
        """Get best metrics achieved."""
        return self.best_metrics
    
    def suggest_hyperparameters(self, performance_data: Dict[str, float]) -> Dict[str, Any]:
        """Suggest hyperparameter adjustments based on performance."""
        suggestions = {}
        
        current_quality = performance_data.get("embedding_quality", 0.0)
        current_loss = performance_data.get("training_loss", 1.0)
        
        # Learning rate suggestions
        if current_loss > 0.5:
            suggestions["learning_rate"] = self.config.learning_rate * 0.5
        elif current_quality < 0.3:
            suggestions["learning_rate"] = self.config.learning_rate * 1.5
        
        # Batch size suggestions
        if current_quality < 0.4:
            suggestions["batch_size"] = min(64, self.config.batch_size * 2)
        
        # Strategy suggestions
        if current_quality < 0.3:
            if self.config.optimization_strategy.value == "contrastive":
                suggestions["optimization_strategy"] = "triplet"
            else:
                suggestions["optimization_strategy"] = "contrastive"
        
        return suggestions