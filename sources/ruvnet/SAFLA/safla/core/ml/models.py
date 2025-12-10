"""
Neural embedding models for the SAFLA ML system.

This module contains the transformer-based embedding models with multi-modal
support and fine-tuning capabilities.
"""

import torch
import torch.nn as nn
from typing import Dict
from transformers import AutoModel, AutoTokenizer

from .base import EmbeddingConfig, EmbeddingType


class TransformerEmbeddingModel(nn.Module):
    """Transformer-based embedding model with fine-tuning capabilities."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config
        self.device = self._get_device()
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Projection layers for different modalities
        self.text_projection = nn.Linear(self.transformer.config.hidden_size, config.embedding_dim)
        self.image_projection = nn.Linear(2048, config.embedding_dim)  # Assuming ResNet features
        self.audio_projection = nn.Linear(1024, config.embedding_dim)  # Assuming audio features
        
        # Multi-modal fusion layers
        if config.multi_modal_fusion == "attention":
            self.fusion_attention = nn.MultiheadAttention(config.embedding_dim, num_heads=8)
        elif config.multi_modal_fusion == "cross_modal":
            self.cross_modal_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(config.embedding_dim, nhead=8),
                num_layers=2
            )
        
        # Optimization layers
        self.contrastive_head = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.classification_head = nn.Linear(config.embedding_dim, 1000)  # For supervised learning
        
        self.to(self.device)
    
    def _get_device(self) -> torch.device:
        """Get appropriate device for computation."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)
    
    def forward(self, inputs: Dict[str, torch.Tensor], modality: EmbeddingType = EmbeddingType.TEXT) -> Dict[str, torch.Tensor]:
        """Forward pass for embedding generation."""
        if modality == EmbeddingType.TEXT:
            return self._forward_text(inputs)
        elif modality == EmbeddingType.MULTIMODAL:
            return self._forward_multimodal(inputs)
        else:
            raise NotImplementedError(f"Modality {modality} not implemented")
    
    def _forward_text(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for text embeddings."""
        outputs = self.transformer(**inputs)
        
        # Use CLS token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Project to target dimension
        embeddings = self.text_projection(embeddings)
        
        # Normalize if configured
        if self.config.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return {
            "embeddings": embeddings,
            "attention_weights": outputs.attentions[-1] if outputs.attentions else None,
            "hidden_states": outputs.last_hidden_state
        }
    
    def _forward_multimodal(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-modal embeddings."""
        modal_embeddings = []
        attention_weights = None
        
        # Process each modality
        for modality_name, modality_inputs in inputs.items():
            if modality_name == "text":
                text_outputs = self._forward_text(modality_inputs)
                modal_embeddings.append(text_outputs["embeddings"])
                attention_weights = text_outputs.get("attention_weights")
            elif modality_name == "image":
                # Project image features
                image_embeddings = self.image_projection(modality_inputs)
                if self.config.normalize_embeddings:
                    image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
                modal_embeddings.append(image_embeddings)
            elif modality_name == "audio":
                # Project audio features
                audio_embeddings = self.audio_projection(modality_inputs)
                if self.config.normalize_embeddings:
                    audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
                modal_embeddings.append(audio_embeddings)
        
        # Fuse modalities
        if len(modal_embeddings) == 1:
            fused_embeddings = modal_embeddings[0]
        else:
            fused_embeddings = self._fuse_modalities(modal_embeddings)
        
        return {
            "embeddings": fused_embeddings,
            "attention_weights": attention_weights,
            "modal_embeddings": modal_embeddings
        }
    
    def _fuse_modalities(self, modal_embeddings: list) -> torch.Tensor:
        """Fuse embeddings from different modalities."""
        if self.config.multi_modal_fusion == "concatenation":
            return torch.cat(modal_embeddings, dim=1)
        elif self.config.multi_modal_fusion == "attention":
            # Use attention-based fusion
            stacked = torch.stack(modal_embeddings, dim=1)  # [batch, num_modalities, embed_dim]
            attended, _ = self.fusion_attention(stacked, stacked, stacked)
            return attended.mean(dim=1)  # Average across modalities
        elif self.config.multi_modal_fusion == "cross_modal":
            # Use cross-modal transformer
            stacked = torch.stack(modal_embeddings, dim=1)
            transformed = self.cross_modal_transformer(stacked.transpose(0, 1))
            return transformed.transpose(0, 1).mean(dim=1)
        else:
            # Default: average fusion
            return torch.stack(modal_embeddings, dim=1).mean(dim=1)
    
    def get_contrastive_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get features for contrastive learning."""
        return self.contrastive_head(embeddings)
    
    def get_classification_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get classification logits for supervised learning."""
        return self.classification_head(embeddings)
    
    def freeze_transformer(self):
        """Freeze transformer parameters for fine-tuning only projection layers."""
        for param in self.transformer.parameters():
            param.requires_grad = False
    
    def unfreeze_transformer(self):
        """Unfreeze transformer parameters for full fine-tuning."""
        for param in self.transformer.parameters():
            param.requires_grad = True
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params
        }
    
    def save_checkpoint(self, path: str, metadata: Dict = None):
        """Save model checkpoint with metadata."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'metadata': metadata or {},
            'model_size': self.get_model_size()
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint.get('metadata', {})


class MultiModalProjector(nn.Module):
    """Specialized projector for multi-modal embeddings."""
    
    def __init__(self, input_dims: Dict[str, int], output_dim: int):
        super().__init__()
        self.projectors = nn.ModuleDict()
        
        for modality, input_dim in input_dims.items():
            self.projectors[modality] = nn.Sequential(
                nn.Linear(input_dim, output_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim)
            )
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Project inputs from different modalities to common space."""
        projected = {}
        for modality, features in inputs.items():
            if modality in self.projectors:
                projected[modality] = self.projectors[modality](features)
        return projected


class ContrastiveLoss(nn.Module):
    """Contrastive loss for embedding learning."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Compute contrastive loss."""
        batch_size = embeddings.size(0)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create labels for contrastive learning
        if labels is None:
            # Self-supervised: positive pairs are augmented versions
            labels = torch.arange(batch_size // 2, device=embeddings.device)
            labels = labels.repeat(2)
        
        # Mask out self-similarity
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        similarity_matrix.masked_fill_(mask, float('-inf'))
        
        # Compute loss
        loss = self.criterion(similarity_matrix, labels)
        return loss


class TripletLoss(nn.Module):
    """Triplet loss for embedding learning."""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss."""
        return self.triplet_loss(anchor, positive, negative)