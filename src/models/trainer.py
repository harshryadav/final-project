"""
Model Training Module for Stock Forecasting

Training pipeline for the Transformer model.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import numpy as np
from typing import Tuple, Dict, Any
import time

from .transformer import create_transformer

class ModelTrainer:
    """
    Trainer for Transformer-based stock prediction.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the trainer
        
        Args:
            verbose: Whether to print training progress
        """
        self.verbose = verbose
        self.last_model_config = None
    
    def create_model(self, 
                     input_shape: Tuple[int, int],
                     config: Dict[str, Any] = None) -> keras.Model:
        """
        Create and configure the Transformer model
        
        Args:
            input_shape: (sequence_length, num_features)
            config: Model configuration dictionary
            
        Returns:
            Configured Transformer model
        """
        # Configuration for stock prediction
        default_config = {
            'seq_len': input_shape[0],
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 4,
            'dff': 512,
            'dropout_rate': 0.1
        }
        
        # Update with user config
        if config:
            default_config.update(config)
        
        # Store config for later use
        self.last_model_config = default_config.copy()
        
        if self.verbose:
            print("Creating Transformer model:")
            for key, value in default_config.items():
                print(f"   {key}: {value}")
        
        # Create model
        model = create_transformer(input_shape, default_config)
        
        # Build the model with sample data
        sample_input = tf.random.normal((1, input_shape[0], input_shape[1]))
        _ = model(sample_input)
        
        if self.verbose:
            print(f"Model created with {model.count_params():,} parameters")
        
        return model
    
    def compile_model(self, model: keras.Model, learning_rate: float = 1e-3) -> None:
        """
        Compile the model with optimizer and loss function
        
        Args:
            model: Transformer model to compile
            learning_rate: Learning rate for optimizer
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean squared error for regression
            metrics=['mae']  # Mean absolute error
        )
        
        if self.verbose:
            print(f"Model compiled with Adam optimizer (lr={learning_rate})")
    
    def create_callbacks(self, 
                        model_save_path: str,
                        patience: int = 10) -> list:
        """
        Create training callbacks
        
        Args:
            model_save_path: Path to save the best model
            patience: Early stopping patience
            
        Returns:
            List of callbacks
        """
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1 if self.verbose else 0
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1 if self.verbose else 0
            ),
            
            # Save best model weights
            callbacks.ModelCheckpoint(
                filepath=f"{model_save_path}.weights.h5",
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1 if self.verbose else 0
            )
        ]
        
        return callbacks_list
    
    def train(self,
              model: keras.Model,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32,
              model_save_path: str = "models/transformer_model") -> Dict[str, Any]:
        """
        Train the Transformer model
        
        Args:
            model: Transformer model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Training batch size
            model_save_path: Path to save trained model
            
        Returns:
            Dictionary containing training history and metadata
        """
        if self.verbose:
            print("\nStarting training...")
            print(f"   Training samples: {len(X_train):,}")
            print(f"   Validation samples: {len(X_val):,}")
            print(f"   Batch size: {batch_size}")
            print(f"   Max epochs: {epochs}")
        
        # Ensure save directory exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Create callbacks
        callback_list = self.create_callbacks(model_save_path)
        
        # Start training
        start_time = time.time()
        
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1 if self.verbose else 0,
            shuffle=True
        )
        
        training_time = time.time() - start_time
        
        if self.verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Best model saved to: {model_save_path}")
        
        # Save final model weights and config
        model.save_weights(f"{model_save_path}.weights.h5")
        
        # Save model configuration for reconstruction
        import json
        model_config = {
            'input_shape': (X_train.shape[1], X_train.shape[2]),
            'config': self.last_model_config
        }
        with open(f"{model_save_path}_config.json", 'w') as f:
            json.dump(model_config, f)
        
        return {
            'history': history.history,
            'training_time': training_time,
            'model_path': model_save_path,
            'final_val_loss': min(history.history['val_loss']),
            'epochs_trained': len(history.history['loss'])
        }
    
    def load_model(self, model_path: str) -> keras.Model:
        """
        Load a trained model from weights and config
        
        Args:
            model_path: Base path to saved model (without extension)
            
        Returns:
            Loaded Transformer model
        """
        import json
        
        # Load model configuration
        config_path = f"{model_path}_config.json"
        weights_path = f"{model_path}.weights.h5"
        
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Recreate the model
        model = self.create_model(
            input_shape=model_config['input_shape'],
            config=model_config['config']
        )
        
        # Load the weights
        model.load_weights(weights_path)
        
        if self.verbose:
            print(f"Model loaded from: {weights_path}")
        
        return model
    
    def predict(self, 
                model: keras.Model, 
                X: np.ndarray,
                batch_size: int = 32) -> np.ndarray:
        """
        Make predictions with the trained model
        
        Args:
            model: Trained Transformer model
            X: Input features
            batch_size: Prediction batch size
            
        Returns:
            Predictions array
        """
        predictions = model.predict(X, batch_size=batch_size, verbose=0)
        
        if self.verbose:
            print(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def evaluate(self,
                 model: keras.Model,
                 X_test: np.ndarray,
                 y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred = self.predict(model, X_test)
        
        # Calculate metrics
        mae = np.mean(np.abs(y_test - y_pred.flatten()))
        mse = np.mean((y_test - y_pred.flatten()) ** 2)
        rmse = np.sqrt(mse)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
        
        if self.verbose:
            print("\nModel Evaluation:")
            print(f"   MAE:  {mae:.4f}")
            print(f"   MSE:  {mse:.4f}")
            print(f"   RMSE: {rmse:.4f}")
        
        return metrics

# For backward compatibility
StockModelTrainer = ModelTrainer 