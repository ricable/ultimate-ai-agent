#!/usr/bin/env python3
"""
Regression and Binary Classification Demo using NeuralFlow

This demo shows various use cases including regression, binary classification,
and multi-class classification with different architectures.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import neuralflow as nf
from neuralflow.utils import data_utils, visualization


def regression_demo():
    """Demonstrate regression with neural networks."""
    print("\n" + "=" * 60)
    print("REGRESSION DEMO: Polynomial Function Approximation")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic regression data...")
    np.random.seed(42)
    
    # Create non-linear function: y = x^2 + sin(2x) + noise
    X = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_true = X**2 + 2*np.sin(2*X) + 0.5
    y = y_true + np.random.normal(0, 0.3, X.shape)
    
    # Split data
    X_train, X_test, y_train, y_test = data_utils.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Feature dimension: {X_train.shape[1]}")
    
    # Create model
    print("\n2. Creating regression model...")
    model = nf.Sequential([
        nf.layers.Dense(64, activation='relu'),
        nf.layers.Dense(32, activation='relu'),
        nf.layers.Dense(16, activation='relu'),
        nf.layers.Dense(1)  # No activation for regression
    ])
    
    # Compile with MSE loss
    print("\n3. Compiling model...")
    model.compile(
        optimizer=nf.optimizers.Adam(learning_rate=0.01),
        loss='mse'
    )
    
    # Train model
    print("\n4. Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=0  # Silent training
    )
    
    # Evaluate
    print("\n5. Evaluating model...")
    test_loss = model.evaluate(X_test, y_test)
    print(f"   Test MSE: {test_loss['loss']:.4f}")
    print(f"   Test RMSE: {np.sqrt(test_loss['loss']):.4f}")
    
    # Make predictions
    print("\n6. Sample predictions:")
    sample_indices = [0, 40, 80, 120, 160]
    for idx in sample_indices[:5]:
        pred = model.predict(X[idx:idx+1])
        print(f"   X={X[idx,0]:.2f}, True={y[idx,0]:.2f}, Predicted={pred[0,0]:.2f}")
    
    # Show training history
    print("\n7. Training History:")
    print(visualization.plot_history(history, metrics=['loss']))
    
    return model, history


def binary_classification_demo():
    """Demonstrate binary classification."""
    print("\n" + "=" * 60)
    print("BINARY CLASSIFICATION DEMO: Circle Dataset")
    print("=" * 60)
    
    # Generate circular dataset
    print("\n1. Generating circular classification data...")
    np.random.seed(42)
    
    # Generate points
    n_samples = 500
    radius = np.random.uniform(0, 2, n_samples)
    angle = np.random.uniform(0, 2*np.pi, n_samples)
    
    X = np.column_stack([
        radius * np.cos(angle),
        radius * np.sin(angle)
    ])
    
    # Label based on distance from origin
    y = (radius < 1.0).astype(int)
    
    # Add some noise
    noise_indices = np.random.choice(n_samples, size=int(0.05*n_samples), replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    # Split data
    X_train, X_test, y_train, y_test = data_utils.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Class distribution: {np.bincount(y_train)}")
    
    # Create model
    print("\n2. Creating binary classification model...")
    model = nf.Sequential([
        nf.layers.Dense(16, activation='relu'),
        nf.layers.Dense(8, activation='relu'),
        nf.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile
    print("\n3. Compiling model...")
    model.compile(
        optimizer=nf.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("\n4. Training model...")
    history = model.fit(
        X_train, y_train.reshape(-1, 1),
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test.reshape(-1, 1)),
        verbose=0
    )
    
    # Evaluate
    print("\n5. Evaluating model...")
    test_metrics = model.evaluate(X_test, y_test.reshape(-1, 1))
    print(f"   Test Loss: {test_metrics['loss']:.4f}")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.2%}")
    
    # Decision boundary visualization (ASCII)
    print("\n6. Decision Boundary Visualization:")
    print("   " + "-" * 40)
    
    # Create grid
    grid_size = 20
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Plot decision boundary
    for i in range(grid_size):
        for j in range(grid_size):
            x_coord = x_min + (x_max - x_min) * j / grid_size
            y_coord = y_max - (y_max - y_min) * i / grid_size
            
            point = np.array([[x_coord, y_coord]])
            pred = model.predict(nf.core.Tensor(point))[0, 0]
            
            if pred > 0.5:
                grid[i][j] = '+'
            else:
                grid[i][j] = '-'
    
    # Print grid
    for row in grid:
        print("   " + ''.join(row))
    
    print("   " + "-" * 40)
    print("   Legend: '+' = Class 1, '-' = Class 0")
    
    return model, history


def multiclass_spiral_demo():
    """Demonstrate multi-class classification on spiral dataset."""
    print("\n" + "=" * 60)
    print("MULTI-CLASS CLASSIFICATION DEMO: Spiral Dataset")
    print("=" * 60)
    
    # Generate spiral data
    print("\n1. Generating spiral dataset...")
    X, y = data_utils.generate_spiral_data(n_points=200, n_classes=3, noise=0.2)
    
    # Split data
    X_train, X_test, y_train, y_test = data_utils.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to categorical
    y_train_cat = data_utils.to_categorical(y_train, num_classes=3)
    y_test_cat = data_utils.to_categorical(y_test, num_classes=3)
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Number of classes: 3")
    print(f"   Features: {X_train.shape[1]}")
    
    # Create model with more capacity for complex pattern
    print("\n2. Creating deep model for spiral classification...")
    model = nf.Sequential([
        nf.layers.Dense(128, activation='relu'),
        nf.layers.BatchNormalization(),
        nf.layers.Dense(64, activation='relu'),
        nf.layers.Dropout(0.3),
        nf.layers.Dense(32, activation='relu'),
        nf.layers.Dense(3, activation='softmax')
    ])
    
    # Compile
    print("\n3. Compiling model...")
    model.compile(
        optimizer=nf.optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("\n4. Training model...")
    history = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test_cat),
        verbose=0
    )
    
    # Evaluate
    print("\n5. Evaluating model...")
    test_metrics = model.evaluate(X_test, y_test_cat)
    print(f"   Test Loss: {test_metrics['loss']:.4f}")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.2%}")
    
    # Show predictions
    print("\n6. Sample predictions:")
    predictions = model.predict(X_test[:10])
    pred_classes = np.argmax(predictions, axis=-1)
    true_classes = y_test[:10]
    
    print(visualization.visualize_predictions(true_classes, pred_classes, n_samples=10))
    
    # Confusion matrix
    print("\n7. Confusion Matrix:")
    all_predictions = model.predict(X_test)
    print(visualization.confusion_matrix(y_test, all_predictions, n_classes=3))
    
    return model, history


def autoencoder_demo():
    """Demonstrate autoencoder for dimensionality reduction."""
    print("\n" + "=" * 60)
    print("AUTOENCODER DEMO: Dimensionality Reduction")
    print("=" * 60)
    
    # Generate high-dimensional data
    print("\n1. Generating high-dimensional data...")
    np.random.seed(42)
    
    # Create data with intrinsic low-dimensional structure
    n_samples = 500
    latent = np.random.randn(n_samples, 2)  # 2D latent space
    
    # Project to high dimensions with non-linear transformation
    projection = np.random.randn(2, 10)
    X = np.tanh(latent @ projection) + 0.1 * np.random.randn(n_samples, 10)
    
    # Normalize
    X = data_utils.normalize(X)
    
    print(f"   Samples: {X.shape[0]}")
    print(f"   Original dimensions: {X.shape[1]}")
    print(f"   Target latent dimensions: 2")
    
    # Create autoencoder
    print("\n2. Creating autoencoder model...")
    
    # Encoder
    encoder_layers = [
        nf.layers.Dense(8, activation='relu'),
        nf.layers.Dense(4, activation='relu'),
        nf.layers.Dense(2, activation='linear')  # Latent space
    ]
    
    # Decoder
    decoder_layers = [
        nf.layers.Dense(4, activation='relu'),
        nf.layers.Dense(8, activation='relu'),
        nf.layers.Dense(10, activation='linear')  # Reconstruction
    ]
    
    # Full autoencoder
    autoencoder = nf.Sequential(encoder_layers + decoder_layers)
    
    # Compile
    print("\n3. Compiling autoencoder...")
    autoencoder.compile(
        optimizer=nf.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Train
    print("\n4. Training autoencoder...")
    history = autoencoder.fit(
        X, X,  # Input and target are the same
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate reconstruction
    print("\n5. Evaluating reconstruction...")
    reconstructed = autoencoder.predict(X[:5])
    
    print("   Sample reconstructions:")
    for i in range(3):
        original = X[i][:5]  # Show first 5 dimensions
        recon = reconstructed[i][:5]
        print(f"   Original: [{', '.join(f'{x:.3f}' for x in original)}...]")
        print(f"   Reconstructed: [{', '.join(f'{x:.3f}' for x in recon)}...]")
        mse = np.mean((X[i] - reconstructed[i])**2)
        print(f"   MSE: {mse:.4f}\n")
    
    # Extract encoder
    print("\n6. Extracting learned features...")
    encoder = nf.Sequential(encoder_layers)
    encoder.layers = autoencoder.layers[:3]  # First 3 layers
    
    # Get latent representations
    latent_representations = encoder.predict(X[:10])
    
    print("   Latent representations (2D):")
    for i in range(5):
        print(f"   Sample {i}: [{latent_representations[i, 0]:.3f}, {latent_representations[i, 1]:.3f}]")
    
    return autoencoder, history


def main():
    """Run all demos."""
    print("=" * 60)
    print("NEURALFLOW COMPREHENSIVE DEMO SUITE")
    print("=" * 60)
    
    demos = [
        ("Regression", regression_demo),
        ("Binary Classification", binary_classification_demo),
        ("Multi-class Classification", multiclass_spiral_demo),
        ("Autoencoder", autoencoder_demo)
    ]
    
    results = {}
    
    for name, demo_func in demos:
        print(f"\n\nRunning {name} Demo...")
        try:
            model, history = demo_func()
            results[name] = {"status": "success", "model": model, "history": history}
        except Exception as e:
            print(f"\nError in {name} demo: {e}")
            results[name] = {"status": "failed", "error": str(e)}
    
    # Summary
    print("\n\n" + "=" * 60)
    print("DEMO SUITE SUMMARY")
    print("=" * 60)
    
    for name, result in results.items():
        status = "✓" if result["status"] == "success" else "✗"
        print(f"{status} {name}: {result['status']}")
    
    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETED!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()