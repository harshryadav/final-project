"""
Transformer Model for Stock Prediction

A clean implementation of a Transformer model for time series forecasting.
Based on the "Attention Is All You Need" paper adapted for stock prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class MultiHeadAttention(layers.Layer):
    """
    Multi-Head Attention mechanism
    """
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split into multiple heads"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        
        # Linear transformations
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.depth, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.dense(output)

class TransformerBlock(layers.Layer):
    """
    Transformer block - one attention + one feed-forward
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.att = MultiHeadAttention(d_model, num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=None):
        # Attention block
        attn_output = self.att(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward block
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(layers.Layer):
    """
    Positional encoding for transformer input
    """
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pos_encoding = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.Variable(
            pos_encoding[np.newaxis, ...].astype(np.float32),
            trainable=False
        )
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

class Transformer(keras.Model):
    """
    Transformer Model for Stock Prediction
    
    Clean implementation for time series forecasting
    """
    def __init__(self, 
                 seq_len=60,
                 d_model=128,
                 num_heads=8,
                 num_layers=4,
                 dff=512,
                 input_features=4,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input processing
        self.input_projection = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(seq_len, d_model)
        self.dropout = layers.Dropout(dropout_rate)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Output layers
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.final_dense = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1)  # Predict single value
    
    def call(self, x, training=None):
        # Input processing
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        # Global average pooling and output
        x = self.global_avg_pool(x)
        x = self.final_dense(x)
        return self.output_layer(x)
    
    def get_config(self):
        return {
            'seq_len': self.seq_len,
            'd_model': self.d_model,
            'num_heads': self.transformer_blocks[0].att.num_heads if self.transformer_blocks else 8,
            'num_layers': len(self.transformer_blocks),
            'dropout_rate': 0.1
        }

def create_transformer(input_shape, config=None):
    """
    Factory function to create a transformer model
    
    Args:
        input_shape: (seq_len, features)
        config: Optional configuration dict
    
    Returns:
        Configured Transformer model
    """
    default_config = {
        'seq_len': input_shape[0],
        'd_model': 128,
        'num_heads': 8,
        'num_layers': 4,
        'dff': 512,
        'input_features': input_shape[1],
        'dropout_rate': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return Transformer(**default_config)

# Demo function
if __name__ == "__main__":
    print("Transformer Demo")
    
    # Create model
    model = create_transformer((30, 4))
    
    # Test with random data
    x = tf.random.normal((32, 30, 4))  # batch_size, seq_len, features
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {model.count_params():,}")
    
    print("\nTransformer model created successfully!") 