#!/usr/bin/env python3
"""
Image Classification Demo using NeuralFlow

This demo shows how to build and train a convolutional neural network
for image classification using the NeuralFlow library.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import neuralflow as nf
from neuralflow.utils import data_utils, visualization


def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """Create a CNN model for image classification."""
    model = nf.Sequential([
        # First convolutional block
        nf.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        nf.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        nf.layers.MaxPool2D(pool_size=2),
        nf.layers.Dropout(0.25),
        
        # Second convolutional block
        nf.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        nf.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        nf.layers.MaxPool2D(pool_size=2),
        nf.layers.Dropout(0.25),
        
        # Flatten and dense layers
        nf.layers.Dense(128, activation='relu'),
        nf.layers.Dropout(0.5),
        nf.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_simple_model(input_shape=(784,), num_classes=10):
    """Create a simple fully connected model."""
    model = nf.Sequential([
        nf.layers.Dense(256, activation='relu'),
        nf.layers.Dropout(0.3),
        nf.layers.Dense(128, activation='relu'),
        nf.layers.Dropout(0.3),
        nf.layers.Dense(64, activation='relu'),
        nf.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def main():
    """Run the image classification demo."""
    print("=" * 60)
    print("NEURALFLOW IMAGE CLASSIFICATION DEMO")
    print("=" * 60)
    
    # Load sample data
    print("\n1. Loading sample MNIST-like data...")
    X_train, y_train, X_test, y_test = data_utils.load_mnist_sample()
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Image shape: {X_train.shape[1:]}")
    print(f"   Number of classes: {len(np.unique(y_train))}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    
    # Flatten images for simple model
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Convert labels to one-hot encoding
    y_train_cat = data_utils.to_categorical(y_train, num_classes=10)
    y_test_cat = data_utils.to_categorical(y_test, num_classes=10)
    
    print("   ✓ Data flattened and normalized")
    print("   ✓ Labels converted to one-hot encoding")
    
    # Create and compile model
    print("\n3. Creating neural network model...")
    model = create_simple_model(input_shape=(784,), num_classes=10)
    
    print("\n   Model Architecture:")
    print("   " + "-" * 40)
    for i, layer in enumerate(model.layers):
        print(f"   Layer {i+1}: {layer}")
    
    # Compile model
    print("\n4. Compiling model...")
    model.compile(
        optimizer=nf.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("   ✓ Optimizer: Adam (lr=0.001)")
    print("   ✓ Loss: Categorical Crossentropy")
    print("   ✓ Metrics: Accuracy")
    
    # Train model
    print("\n5. Training model...")
    print("   " + "-" * 40)
    
    history = model.fit(
        X_train_flat, y_train_cat,
        epochs=10,
        batch_size=32,
        validation_data=(X_test_flat, y_test_cat),
        verbose=1
    )
    
    # Evaluate model
    print("\n6. Evaluating model on test set...")
    test_metrics = model.evaluate(X_test_flat, y_test_cat)
    
    print(f"\n   Final Test Loss: {test_metrics['loss']:.4f}")
    print(f"   Final Test Accuracy: {test_metrics['accuracy']:.2%}")
    
    # Make predictions
    print("\n7. Making predictions on sample data...")
    n_samples = 10
    sample_indices = np.random.choice(X_test.shape[0], n_samples, replace=False)
    X_samples = X_test_flat[sample_indices]
    y_samples = y_test[sample_indices]
    
    predictions = model.predict(X_samples)
    predicted_classes = np.argmax(predictions, axis=-1)
    
    print(visualization.visualize_predictions(y_samples, predicted_classes, n_samples))
    
    # Visualize training history
    print("\n8. Training History:")
    print(visualization.plot_history(history, metrics=['loss', 'accuracy']))
    
    # Show confusion matrix
    print("\n9. Confusion Matrix on Test Set:")
    all_predictions = model.predict(X_test_flat)
    print(visualization.confusion_matrix(y_test, all_predictions))
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Advanced CNN Demo (commented out for faster execution)
    print("\n\nNote: For a more advanced CNN demo, uncomment the CNN code below:")
    print("```python")
    print("# Create CNN model")
    print("cnn_model = create_cnn_model(input_shape=(28, 28, 1), num_classes=10)")
    print("cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])")
    print("# Train on original image data (not flattened)")
    print("cnn_history = cnn_model.fit(X_train, y_train_cat, epochs=5, batch_size=32)")
    print("```")
    
    return model, history


if __name__ == "__main__":
    model, history = main()