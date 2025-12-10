#!/usr/bin/env python3
"""
Text Generation Demo using NeuralFlow

This demo shows how to build and train a recurrent neural network
for text generation using the NeuralFlow library.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import neuralflow as nf
from neuralflow.utils import data_utils


class TextGenerator:
    """Text generation model using LSTM."""
    
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=256):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.char_to_idx = {}
        self.idx_to_char = {}
    
    def build_model(self, sequence_length):
        """Build the LSTM model for text generation."""
        self.model = nf.Sequential([
            # Embedding layer (simplified as Dense for this demo)
            nf.layers.Dense(self.embedding_dim, activation='relu'),
            
            # LSTM layers
            nf.layers.LSTM(self.lstm_units, return_sequences=True),
            nf.layers.Dropout(0.3),
            nf.layers.LSTM(self.lstm_units // 2),
            nf.layers.Dropout(0.3),
            
            # Output layer
            nf.layers.Dense(self.vocab_size, activation='softmax')
        ])
        
        return self.model
    
    def create_sequences(self, text, sequence_length=40):
        """Create training sequences from text."""
        # Create character mappings
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Convert text to indices
        text_indices = [self.char_to_idx[ch] for ch in text]
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(text_indices) - sequence_length):
            seq = text_indices[i:i + sequence_length]
            target = text_indices[i + sequence_length]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def generate_text(self, seed_text, length=100, temperature=1.0):
        """Generate text from a seed."""
        generated = seed_text
        seed_indices = [self.char_to_idx.get(ch, 0) for ch in seed_text[-40:]]
        
        for _ in range(length):
            # Prepare input
            x = np.array([seed_indices[-40:]])
            
            # One-hot encode
            x_encoded = np.zeros((1, 40, self.vocab_size))
            for t, idx in enumerate(x[0]):
                x_encoded[0, t, idx] = 1
            
            # Predict next character
            predictions = self.model.predict(nf.core.Tensor(x_encoded))
            
            # Sample from predictions
            preds = predictions[0]
            preds = np.log(preds + 1e-8) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            
            next_idx = np.random.choice(self.vocab_size, p=preds)
            next_char = self.idx_to_char[next_idx]
            
            generated += next_char
            seed_indices.append(next_idx)
        
        return generated


def create_rnn_model(vocab_size, sequence_length):
    """Create a simple RNN model for text generation."""
    model = nf.Sequential([
        # Input processing
        nf.layers.Dense(128, activation='relu'),
        
        # Recurrent layers  
        nf.layers.Dense(256, activation='relu'),
        nf.layers.Dropout(0.3),
        nf.layers.Dense(128, activation='relu'),
        nf.layers.Dropout(0.3),
        
        # Output layer
        nf.layers.Dense(vocab_size, activation='softmax')
    ])
    
    return model


def main():
    """Run the text generation demo."""
    print("=" * 60)
    print("NEURALFLOW TEXT GENERATION DEMO")
    print("=" * 60)
    
    # Sample text data
    print("\n1. Preparing text data...")
    sample_text = """
    In the realm of artificial intelligence, neural networks have emerged as powerful tools.
    They learn patterns from data through layers of interconnected neurons.
    Deep learning models can process images, understand language, and even generate text.
    The future of AI is bright with endless possibilities for innovation and discovery.
    Machine learning continues to transform how we interact with technology.
    From computer vision to natural language processing, AI is reshaping our world.
    Neural networks mimic the human brain's ability to learn and adapt.
    Training these models requires data, computational power, and clever algorithms.
    The applications of deep learning span across industries and domains.
    As we advance, ethical considerations become increasingly important.
    """
    
    # Clean text
    sample_text = sample_text.lower().strip()
    sample_text = ' '.join(sample_text.split())  # Normalize whitespace
    
    print(f"   Text length: {len(sample_text)} characters")
    print(f"   Unique characters: {len(set(sample_text))}")
    
    # Create character mappings
    chars = sorted(list(set(sample_text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Sample vocabulary: {chars[:10]}...")
    
    # Create training sequences
    print("\n2. Creating training sequences...")
    sequence_length = 40
    
    sequences = []
    targets = []
    
    for i in range(len(sample_text) - sequence_length):
        seq = sample_text[i:i + sequence_length]
        target = sample_text[i + sequence_length]
        
        # Convert to indices
        seq_indices = [char_to_idx[ch] for ch in seq]
        target_idx = char_to_idx[target]
        
        sequences.append(seq_indices)
        targets.append(target_idx)
    
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"   Number of sequences: {len(sequences)}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Input shape: {X.shape}")
    
    # One-hot encode
    print("\n3. One-hot encoding data...")
    X_encoded = np.zeros((X.shape[0], sequence_length, vocab_size))
    for i, sequence in enumerate(X):
        for t, char_idx in enumerate(sequence):
            X_encoded[i, t, char_idx] = 1
    
    y_encoded = data_utils.to_categorical(y, num_classes=vocab_size)
    
    # Flatten sequences for simple model
    X_flat = X_encoded.reshape(X_encoded.shape[0], -1)
    
    print(f"   Encoded input shape: {X_flat.shape}")
    print(f"   Encoded target shape: {y_encoded.shape}")
    
    # Create and compile model
    print("\n4. Creating neural network model...")
    model = create_rnn_model(vocab_size, sequence_length)
    
    print("\n   Model Architecture:")
    print("   " + "-" * 40)
    for i, layer in enumerate(model.layers):
        print(f"   Layer {i+1}: {layer}")
    
    # Compile model
    print("\n5. Compiling model...")
    model.compile(
        optimizer=nf.optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("\n6. Training model...")
    print("   " + "-" * 40)
    
    # Split data
    split_idx = int(0.9 * len(X_flat))
    X_train, X_val = X_flat[:split_idx], X_flat[split_idx:]
    y_train, y_val = y_encoded[:split_idx], y_encoded[split_idx:]
    
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Generate text
    print("\n7. Generating text...")
    print("   " + "-" * 40)
    
    def generate_text(seed_text, length=200):
        """Generate text from seed."""
        generated = seed_text
        seed = seed_text.lower()
        
        for _ in range(length):
            # Prepare input
            if len(seed) < sequence_length:
                seed = seed.ljust(sequence_length)
            else:
                seed = seed[-sequence_length:]
            
            # Convert to indices and one-hot
            seq_indices = [char_to_idx.get(ch, 0) for ch in seed]
            x = np.zeros((1, sequence_length, vocab_size))
            for t, idx in enumerate(seq_indices):
                x[0, t, idx] = 1
            
            x_flat = x.reshape(1, -1)
            
            # Predict
            predictions = model.predict(nf.core.Tensor(x_flat))
            
            # Sample from predictions
            # Use temperature sampling for variety
            temperature = 0.8
            preds = predictions[0]
            preds = np.log(preds + 1e-8) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            
            next_idx = np.random.choice(vocab_size, p=preds)
            next_char = idx_to_char[next_idx]
            
            generated += next_char
            seed += next_char
        
        return generated
    
    # Generate samples with different seeds
    seeds = [
        "neural networks ",
        "deep learning ",
        "the future ",
        "artificial intelligence "
    ]
    
    for seed in seeds:
        print(f"\n   Seed: '{seed}'")
        generated = generate_text(seed, length=100)
        print(f"   Generated: '{generated}'")
    
    # Show training history
    print("\n8. Training History:")
    print(nf.utils.visualization.plot_history(history, metrics=['loss', 'accuracy']))
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n\nNote: For better results:")
    print("- Use more training data")
    print("- Train for more epochs")
    print("- Use actual LSTM/GRU layers (when available)")
    print("- Implement beam search for text generation")
    print("- Use larger sequence lengths")
    
    return model, history


if __name__ == "__main__":
    model, history = main()