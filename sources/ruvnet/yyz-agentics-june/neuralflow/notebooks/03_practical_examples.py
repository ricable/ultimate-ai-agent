"""
# Practical Examples with NeuralFlow

This notebook demonstrates practical applications of neural networks using NeuralFlow,
including real-world inspired examples and best practices.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import neuralflow as nf
from neuralflow.utils import data_utils, visualization

# %% [markdown]
# ## 1. Sentiment Analysis (Text Classification)

# %%
print("=== Sentiment Analysis Example ===\n")

# Simulate text data with simple word embeddings
vocab_size = 5000
max_sequence_length = 100
embedding_dim = 50

# Create model for sentiment analysis
sentiment_model = nf.Sequential([
    # Embedding simulation
    nf.layers.Dense(embedding_dim, activation='linear'),
    
    # Feature extraction
    nf.layers.Dense(128, activation='relu'),
    nf.layers.Dropout(0.5),
    nf.layers.Dense(64, activation='relu'),
    
    # Classification
    nf.layers.Dense(1, activation='sigmoid')  # Binary: positive/negative
])

sentiment_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Sentiment Analysis Model:")
for i, layer in enumerate(sentiment_model.layers):
    print(f"Layer {i+1}: {layer}")

# Generate sample data
n_samples = 1000
X_text = np.random.randint(0, vocab_size, (n_samples, max_sequence_length))
# One-hot encode (simplified)
X_encoded = np.zeros((n_samples, max_sequence_length * vocab_size))
y_sentiment = np.random.randint(0, 2, (n_samples, 1))  # 0: negative, 1: positive

print(f"\nDataset: {n_samples} reviews")
print(f"Positive reviews: {y_sentiment.sum()}")
print(f"Negative reviews: {n_samples - y_sentiment.sum()}")

# %% [markdown]
# ## 2. Time Series Prediction

# %%
print("\n=== Time Series Forecasting ===\n")

# Generate synthetic time series data
def generate_time_series(n_points=1000):
    time = np.linspace(0, 100, n_points)
    # Combine multiple patterns
    trend = 0.05 * time
    seasonal = 10 * np.sin(0.1 * time)
    noise = np.random.normal(0, 1, n_points)
    series = trend + seasonal + noise
    return series

# Create sequences for training
def create_sequences(data, seq_length=50, pred_length=1):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length - pred_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length:i+seq_length+pred_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Generate data
series = generate_time_series()
X_seq, y_seq = create_sequences(series, seq_length=50)

print(f"Time series length: {len(series)}")
print(f"Training sequences: {X_seq.shape}")
print(f"Prediction targets: {y_seq.shape}")

# Time series model
timeseries_model = nf.Sequential([
    nf.layers.Dense(128, activation='relu'),
    nf.layers.Dropout(0.2),
    nf.layers.Dense(64, activation='relu'),
    nf.layers.Dense(32, activation='relu'),
    nf.layers.Dense(1)  # Predict next value
])

timeseries_model.compile(
    optimizer=nf.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

print("\nTime Series Model Architecture:")
for i, layer in enumerate(timeseries_model.layers):
    print(f"Layer {i+1}: {layer}")

# %% [markdown]
# ## 3. Image Segmentation (Simplified)

# %%
print("\n=== Image Segmentation Example ===\n")

# Simplified U-Net style architecture for segmentation
def create_segmentation_model(input_shape=(128, 128, 3), n_classes=21):
    """Create a simplified segmentation model."""
    model = nf.Sequential([
        # Encoder
        nf.layers.Conv2D(64, 3, padding='same', activation='relu'),
        nf.layers.Conv2D(64, 3, padding='same', activation='relu'),
        nf.layers.MaxPool2D(2),
        
        nf.layers.Conv2D(128, 3, padding='same', activation='relu'),
        nf.layers.Conv2D(128, 3, padding='same', activation='relu'),
        nf.layers.MaxPool2D(2),
        
        # Bottleneck
        nf.layers.Conv2D(256, 3, padding='same', activation='relu'),
        
        # Decoder (simplified - in practice would use UpSampling)
        nf.layers.Conv2D(128, 3, padding='same', activation='relu'),
        nf.layers.Conv2D(64, 3, padding='same', activation='relu'),
        
        # Output
        nf.layers.Conv2D(n_classes, 1, padding='same', activation='softmax')
    ])
    
    return model

seg_model = create_segmentation_model()
print("Segmentation Model (Simplified U-Net):")
print("Encoder -> Bottleneck -> Decoder -> Pixel Classification")
print(f"Total layers: {len(seg_model.layers)}")

# %% [markdown]
# ## 4. Anomaly Detection with Autoencoders

# %%
print("\n=== Anomaly Detection ===\n")

# Generate normal and anomalous data
n_normal = 800
n_anomaly = 200
feature_dim = 20

# Normal data: clustered around certain patterns
normal_data = np.random.randn(n_normal, feature_dim) * 0.5 + \
              np.sin(np.linspace(0, 2*np.pi, feature_dim))

# Anomalous data: different distribution
anomaly_data = np.random.randn(n_anomaly, feature_dim) * 2 + \
               np.random.randint(-5, 5, feature_dim)

# Combine data
X_anomaly = np.vstack([normal_data, anomaly_data])
y_anomaly = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])

print(f"Normal samples: {n_normal}")
print(f"Anomalous samples: {n_anomaly}")
print(f"Features: {feature_dim}")

# Autoencoder for anomaly detection
anomaly_detector = nf.Sequential([
    # Encoder
    nf.layers.Dense(15, activation='relu'),
    nf.layers.Dense(10, activation='relu'),
    nf.layers.Dense(5, activation='relu'),  # Bottleneck
    
    # Decoder
    nf.layers.Dense(10, activation='relu'),
    nf.layers.Dense(15, activation='relu'),
    nf.layers.Dense(feature_dim, activation='linear')  # Reconstruction
])

anomaly_detector.compile(
    optimizer='adam',
    loss='mse'
)

print("\nAnomaly Detection Autoencoder:")
print("Train on normal data only, detect anomalies by reconstruction error")

# %% [markdown]
# ## 5. Multi-Label Classification

# %%
print("\n=== Multi-Label Classification ===\n")

# Example: Document categorization (can belong to multiple categories)
n_samples = 1000
n_features = 100
n_labels = 10

# Generate synthetic multi-label data
X_multilabel = np.random.randn(n_samples, n_features)
# Each sample can have multiple labels
y_multilabel = np.random.randint(0, 2, (n_samples, n_labels))

print(f"Samples: {n_samples}")
print(f"Features: {n_features}")
print(f"Possible labels: {n_labels}")
print(f"Average labels per sample: {y_multilabel.sum(axis=1).mean():.1f}")

# Multi-label model
multilabel_model = nf.Sequential([
    nf.layers.Dense(256, activation='relu'),
    nf.layers.BatchNormalization(),
    nf.layers.Dropout(0.5),
    nf.layers.Dense(128, activation='relu'),
    nf.layers.BatchNormalization(),
    nf.layers.Dropout(0.3),
    nf.layers.Dense(n_labels, activation='sigmoid')  # Sigmoid for each label
])

multilabel_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Binary CE for each label
    metrics=['accuracy']
)

print("\nMulti-Label Model:")
print("Uses sigmoid activation and binary crossentropy")
print("Each output neuron represents one label (independent)")

# %% [markdown]
# ## 6. Transfer Learning Simulation

# %%
print("\n=== Transfer Learning Concept ===\n")

# Simulate a pre-trained feature extractor
pretrained_features = nf.Sequential([
    nf.layers.Conv2D(64, 3, padding='same', activation='relu'),
    nf.layers.Conv2D(64, 3, padding='same', activation='relu'),
    nf.layers.MaxPool2D(2),
    nf.layers.Conv2D(128, 3, padding='same', activation='relu'),
    nf.layers.Conv2D(128, 3, padding='same', activation='relu'),
    nf.layers.MaxPool2D(2),
])

# New task-specific head
new_head = nf.Sequential([
    nf.layers.Dense(256, activation='relu'),
    nf.layers.Dropout(0.5),
    nf.layers.Dense(10, activation='softmax')  # New task: 10 classes
])

print("Transfer Learning Architecture:")
print("1. Pre-trained feature extractor (frozen weights)")
print("2. New classification head (trainable)")
print("\nBenefits:")
print("- Faster training")
print("- Better performance with limited data")
print("- Leverage knowledge from large datasets")

# %% [markdown]
# ## 7. Best Practices Summary

# %%
print("\n=== Best Practices for Production ===\n")

print("1. Data Preprocessing:")
print("   - Always normalize/standardize inputs")
print("   - Handle missing values appropriately")
print("   - Use appropriate encoding for categorical data")
print("   - Split data properly (train/val/test)")

print("\n2. Model Development:")
print("   - Start simple, increase complexity gradually")
print("   - Use regularization (dropout, batch norm)")
print("   - Monitor for overfitting")
print("   - Use appropriate metrics for your task")

print("\n3. Training:")
print("   - Use callbacks for learning rate scheduling")
print("   - Implement early stopping")
print("   - Save best model during training")
print("   - Log experiments for reproducibility")

print("\n4. Evaluation:")
print("   - Use appropriate metrics (not just accuracy)")
print("   - Evaluate on held-out test set")
print("   - Consider cross-validation for small datasets")
print("   - Analyze errors to improve model")

print("\n5. Deployment:")
print("   - Optimize model size if needed")
print("   - Test inference speed")
print("   - Monitor model performance in production")
print("   - Plan for model updates")

# %% [markdown]
# ## Example: Complete Workflow

# %%
print("\n=== Complete Machine Learning Workflow ===\n")

# 1. Load and preprocess data
print("Step 1: Data Preparation")
X, y = data_utils.generate_spiral_data(n_points=300, n_classes=3)
X = data_utils.normalize(X)
y_cat = data_utils.to_categorical(y, 3)
X_train, X_test, y_train, y_test = data_utils.train_test_split(X, y_cat, test_size=0.2)
print(f"✓ Data loaded and preprocessed: {X_train.shape}")

# 2. Create model
print("\nStep 2: Model Creation")
model = nf.Sequential([
    nf.layers.Dense(64, activation='relu'),
    nf.layers.BatchNormalization(),
    nf.layers.Dropout(0.3),
    nf.layers.Dense(32, activation='relu'),
    nf.layers.Dense(3, activation='softmax')
])
print("✓ Model architecture defined")

# 3. Compile
print("\nStep 3: Model Compilation")
model.compile(
    optimizer=nf.optimizers.Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("✓ Model compiled with optimizer and loss")

# 4. Train
print("\nStep 4: Training")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=0
)
print("✓ Model trained successfully")

# 5. Evaluate
print("\nStep 5: Evaluation")
test_metrics = model.evaluate(X_test, y_test)
print(f"✓ Test Accuracy: {test_metrics['accuracy']:.2%}")

# 6. Analyze
print("\nStep 6: Analysis")
predictions = model.predict(X_test)
print(visualization.confusion_matrix(y_test, predictions, n_classes=3))

# %% [markdown]
# ## Summary
# 
# This notebook covered practical examples including:
# - Sentiment analysis for text
# - Time series forecasting
# - Image segmentation concepts
# - Anomaly detection with autoencoders
# - Multi-label classification
# - Transfer learning concepts
# - Best practices for production
# - Complete ML workflow example
# 
# NeuralFlow provides the flexibility to implement these diverse applications
# while maintaining code simplicity and readability.

print("\nNotebook completed successfully!")