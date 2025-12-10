#!/usr/bin/env python3
"""
LSTM Climate Model Training Script

This script trains an LSTM neural network for climate prediction
and exports it to ONNX format for Rust inference.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ClimateDataset(Dataset):
    """Climate time series dataset"""

    def __init__(self, data_path: Path, sequence_length: int = 10):
        self.sequence_length = sequence_length

        # Load data (assuming CSV or numpy format)
        if data_path.suffix == ".npy":
            self.data = np.load(data_path)
        else:
            self.data = np.loadtxt(data_path, delimiter=",")

        # Normalize data
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0) + 1e-8
        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class LSTMClimateModel(nn.Module):
    """LSTM model for climate prediction"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]

        # Predict
        output = self.fc(last_output)
        return output


class Trainer:
    """Training orchestrator"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.history = []

    def train(self, model: nn.Module, train_loader, val_loader=None):
        """Train the model"""

        optimizer = self._create_optimizer(model)
        criterion = nn.MSELoss()
        scheduler = self._create_scheduler(optimizer)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config["epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            val_loss = None
            if val_loader:
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss if val_loss else train_loss)
                else:
                    scheduler.step()

            # Record history
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_stats = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "duration_secs": 0.0,  # Would need timer
            }
            self.history.append(epoch_stats)

            print(
                f"Epoch {epoch+1}/{self.config['epochs']}: "
                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f if val_loss else 'N/A'}, "
                f"LR={current_lr:.6f}"
            )

            # Early stopping
            if val_loss and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            early_stop_patience = self.config.get("early_stopping_patience")
            if early_stop_patience and patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    def _create_optimizer(self, model):
        """Create optimizer from config"""
        optimizer_name = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 1e-5)

        if optimizer_name == "adam":
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(
                model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler from config"""
        scheduler_config = self.config.get("lr_scheduler")

        if not scheduler_config:
            return None

        sched_type = scheduler_config.get("type")

        if sched_type == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 10),
                min_lr=scheduler_config.get("min_lr", 1e-6),
            )
        elif sched_type == "CosineAnnealing":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get("t_max", 100),
                eta_min=scheduler_config.get("eta_min", 0),
            )

        return None


def export_to_onnx(
    model: nn.Module, output_path: Path, input_dim: int, sequence_length: int
):
    """Export model to ONNX format"""
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, sequence_length, input_dim)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Model exported to ONNX: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM climate model")
    parser.add_argument("--data", type=Path, required=True, help="Training data path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", type=Path, required=True, help="Config JSON path")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    print(f"Training with config: {json.dumps(config, indent=2)}")

    # Create dataset
    dataset = ClimateDataset(args.data)

    # Split into train/val
    val_split = config.get("validation_split", 0.2)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
    )

    # Create model
    input_dim = dataset.data.shape[1]
    hidden_dim = config.get("hidden_dims", [64])[0]
    num_layers = config.get("num_layers", 2)
    output_dim = config.get("output_dim", input_dim)
    dropout = config.get("dropout", 0.2)

    model = LSTMClimateModel(input_dim, hidden_dim, num_layers, output_dim, dropout)

    device = torch.device(config.get("device", "cpu"))
    model = model.to(device)

    print(f"Model architecture:\n{model}")

    # Train
    trainer = Trainer(config)
    trainer.train(model, train_loader, val_loader)

    # Save training history
    args.output.mkdir(parents=True, exist_ok=True)
    history_path = args.output / "training_history.json"

    with open(history_path, "w") as f:
        json.dump(trainer.history, f, indent=2)

    # Export to ONNX
    onnx_path = args.output / "lstm_climate.onnx"
    export_to_onnx(model, onnx_path, input_dim, dataset.sequence_length)

    print("Training complete!")


if __name__ == "__main__":
    main()
