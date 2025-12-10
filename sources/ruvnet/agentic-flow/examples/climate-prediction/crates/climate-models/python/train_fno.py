#!/usr/bin/env python3
"""
Fourier Neural Operator (FNO) Training Script

This script trains a Fourier Neural Operator for spatial-temporal
climate prediction and exports to ONNX format.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class SpectralConv2d(nn.Module):
    """2D Fourier layer for FNO"""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def _compl_mul2d(self, input, weights):
        """Complex multiplication"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)


class FNOClimateModel(nn.Module):
    """Fourier Neural Operator for climate prediction"""

    def __init__(
        self, modes: int, width: int, input_dim: int, output_dim: int, num_layers: int = 4
    ):
        super().__init__()

        self.modes = modes
        self.width = width
        self.num_layers = num_layers

        # Input projection
        self.fc0 = nn.Linear(input_dim, width)

        # Fourier layers
        self.spectral_layers = nn.ModuleList()
        self.ws = nn.ModuleList()

        for _ in range(num_layers):
            self.spectral_layers.append(SpectralConv2d(width, width, modes, modes))
            self.ws.append(nn.Conv2d(width, width, 1))

        # Output projection
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # x shape: (batch, channels, height, width)

        # Permute for linear layer
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # Apply Fourier layers
        for spectral_layer, w in zip(self.spectral_layers, self.ws):
            x1 = spectral_layer(x)
            x2 = w(x)
            x = x1 + x2
            x = torch.nn.functional.gelu(x)

        # Output projection
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)

        return x


class ClimateSpatialDataset(Dataset):
    """Spatial-temporal climate dataset"""

    def __init__(self, data_path: Path):
        # Load spatial data (assuming numpy format)
        if data_path.suffix == ".npy":
            self.data = np.load(data_path)
        else:
            raise ValueError("FNO requires numpy data format")

        # Normalize
        self.mean = self.data.mean()
        self.std = self.data.std() + 1e-8
        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.data[idx + 1]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class FNOTrainer:
    """Training orchestrator for FNO"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.history = []

    def train(self, model: nn.Module, train_loader, val_loader=None):
        """Train the FNO model"""

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.get("learning_rate", 0.001),
            weight_decay=self.config.get("weight_decay", 1e-5),
        )

        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10, min_lr=1e-6
        )

        for epoch in range(self.config["epochs"]):
            # Training
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

            # Validation
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
                scheduler.step(val_loss)

            # Record history
            current_lr = optimizer.param_groups[0]["lr"]
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "duration_secs": 0.0,
                }
            )

            print(
                f"Epoch {epoch+1}/{self.config['epochs']}: "
                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f if val_loss else 'N/A'}"
            )


def export_to_onnx(model: nn.Module, output_path: Path, input_shape: tuple):
    """Export FNO model to ONNX"""
    model.eval()

    # Create dummy input (batch, channels, height, width)
    dummy_input = torch.randn(*input_shape)

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

    print(f"FNO model exported to ONNX: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train FNO climate model")
    parser.add_argument("--data", type=Path, required=True, help="Training data path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", type=Path, required=True, help="Config JSON path")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    print(f"Training FNO with config: {json.dumps(config, indent=2)}")

    # Load dataset
    dataset = ClimateSpatialDataset(args.data)

    # Split dataset
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
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Create FNO model
    modes = config.get("extra", {}).get("modes", 12)
    width = config.get("extra", {}).get("width", 64)
    input_dim = config.get("input_dim", 3)
    output_dim = config.get("output_dim", 3)
    num_layers = config.get("num_layers", 4)

    model = FNOClimateModel(modes, width, input_dim, output_dim, num_layers)

    device = torch.device(config.get("device", "cpu"))
    model = model.to(device)

    print(f"FNO Model architecture:\n{model}")

    # Train
    trainer = FNOTrainer(config)
    trainer.train(model, train_loader, val_loader)

    # Save history
    args.output.mkdir(parents=True, exist_ok=True)
    history_path = args.output / "training_history.json"

    with open(history_path, "w") as f:
        json.dump(trainer.history, f, indent=2)

    # Export to ONNX
    onnx_path = args.output / "fno_climate.onnx"

    # Get shape from dataset
    sample, _ = dataset[0]
    input_shape = (1, *sample.shape)

    export_to_onnx(model, onnx_path, input_shape)

    print("FNO training complete!")


if __name__ == "__main__":
    main()
