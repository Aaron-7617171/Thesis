"""
Transformer Model for Regression
A Machine Learning model using Transformer Architecture with:
- 7 Features as inputs
- 2 Hidden Layers (Transformer blocks)
- 1 Output layer (float regression)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    """
    Transformer Block implementing multi-head attention and feed-forward network
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Multi-head attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


def create_transformer_regression_model(
    input_features=7,
    embed_dim=32,
    num_heads=4,
    ff_dim=64,
    num_transformer_blocks=2,
    dropout_rate=0.1,
    mlp_units=[64],
    mlp_dropout=0.1
):
    """
    Create a Transformer model for regression tasks
    
    Parameters:
    -----------
    input_features : int
        Number of input features (default: 7)
    embed_dim : int
        Embedding dimension for the transformer
    num_heads : int
        Number of attention heads
    ff_dim : int
        Hidden layer size in feed-forward network
    num_transformer_blocks : int
        Number of transformer blocks (hidden layers, default: 2)
    dropout_rate : float
        Dropout rate for transformer blocks
    mlp_units : list
        List of hidden units for final MLP layers
    mlp_dropout : float
        Dropout rate for MLP layers
    
    Returns:
    --------
    keras.Model
        Compiled Keras model with Transformer architecture
    """
    
    # Input layer - 7 features
    inputs = layers.Input(shape=(input_features,))
    
    # Reshape to (batch_size, 1, input_features) for sequence processing
    # Transformer expects 3D input: (batch_size, sequence_length, features)
    x = layers.Reshape((1, input_features))(inputs)
    
    # Projection to embedding dimension
    x = layers.Dense(embed_dim)(x)
    
    # Add 2 Transformer blocks (hidden layers)
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)
    
    # Global average pooling to reduce sequence dimension
    x = layers.GlobalAveragePooling1D()(x)
    
    # Add dropout
    x = layers.Dropout(mlp_dropout)(x)
    
    # Additional MLP layers for final processing
    for units in mlp_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    # Output layer - single float value for regression
    outputs = layers.Dense(1)(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate loss and optimizer for regression
    
    Parameters:
    -----------
    model : keras.Model
        The model to compile
    learning_rate : float
        Learning rate for the optimizer
    
    Returns:
    --------
    keras.Model
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae', 'mse']  # Mean Absolute Error and MSE for monitoring
    )
    return model


def main():
    """
    Example usage of the Transformer regression model
    """
    print("=" * 60)
    print("Creating Transformer Model for Regression")
    print("=" * 60)
    
    # Create the model with 7 input features, 2 hidden layers (Transformer blocks)
    model = create_transformer_regression_model(
        input_features=7,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_transformer_blocks=2,  # 2 hidden layers
        dropout_rate=0.1,
        mlp_units=[64],
        mlp_dropout=0.1
    )
    
    # Compile the model
    model = compile_model(model, learning_rate=0.001)
    
    # Display model architecture
    print("\nModel Architecture:")
    print("-" * 60)
    model.summary()
    
    print("\n" + "=" * 60)
    print("Model Configuration:")
    print("-" * 60)
    print(f"Input Features: 7")
    print(f"Hidden Layers: 2 (Transformer blocks)")
    print(f"Output Layer: 1 (float regression)")
    print(f"Total Parameters: {model.count_params():,}")
    print("=" * 60)
    
    # Generate dummy data for testing
    print("\nTesting model with dummy data...")
    X_dummy = np.random.randn(10, 7)  # 10 samples, 7 features
    y_dummy = np.random.randn(10, 1)  # 10 samples, 1 output
    
    # Test prediction
    predictions = model.predict(X_dummy, verbose=0)
    print(f"Dummy input shape: {X_dummy.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:3].flatten()}")
    
    # Test training on dummy data
    print("\nTesting model training on dummy data...")
    history = model.fit(
        X_dummy, 
        y_dummy, 
        epochs=5, 
        batch_size=2, 
        verbose=1,
        validation_split=0.2
    )
    
    print("\n" + "=" * 60)
    print("Model successfully created and tested!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run the main function
    model = main()
