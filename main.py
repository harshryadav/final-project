"""
Stock Price Forecasting with Transformers - Main Pipeline

Implementation of a Transformer model for stock price prediction.

Project Requirements:
- Transformer-based model for time series forecasting
- Historical stock data with technical indicators
- Predict next-day closing price
- Evaluation using MAE, RMSE, MAPE, R²
- Built with TensorFlow/Keras
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from data.data_fetcher import StockDataFetcher
from data.stock_preprocessor import StockDataPreprocessor
from models.trainer import ModelTrainer
from evaluation.metrics import StockForecastingMetrics

class StockForecastingPipeline:
    """
    Pipeline for stock price forecasting with Transformer
    """
    
    def __init__(self, config: dict = None):
        """Initialize the pipeline with configuration"""
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Initialize components
        self.data_fetcher = StockDataFetcher()
        self.preprocessor = None  # Will be configured when needed
        self.trainer = ModelTrainer(verbose=True)
        self.metrics = StockForecastingMetrics()
        
        # Create output directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
    
    def _get_default_config(self) -> dict:
        """Get default configuration optimized for stock prediction"""
        return {
            # Data configuration
            'symbol': 'AAPL',
            'start_date': '2019-01-01',
            'end_date': '2024-01-01',
            
            # Preprocessing
            'sequence_length': 60,  # 60 days as per proposal
            'prediction_horizon': 1,  # Predict next day
            'scaler_type': 'standard',
            'features': ['Open', 'High', 'Low', 'Close', 'Volume'],
            
            # Model configuration (can be extended by team)
            'model_config': {
                'd_model': 128,
                'num_heads': 8,
                'num_layers': 4,
                'dff': 512,
                'dropout_rate': 0.1
            },
            
            # Training
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'patience': 15,
            
            # Evaluation
            'model_save_path': 'models/transformer_stock_model'
        }
    
    def run_complete_pipeline(self):
        """
        Run the complete pipeline from data fetching to evaluation
        """
        print("Starting Stock Forecasting Pipeline")
        print("="*60)
        
        # Step 1: Fetch Data
        print("\nStep 1: Fetching Stock Data")
        raw_data = self.data_fetcher.fetch_stock_data(
            symbol=self.config['symbol'],
            start_date=self.config['start_date'],
            end_date=self.config['end_date']
        )
        print(f"Fetched {len(raw_data)} records for {self.config['symbol']}")
        
        # Step 2: Preprocess Data
        print("\nStep 2: Preprocessing Data")
        # Configure preprocessor
        self.preprocessor = StockDataPreprocessor(
            sequence_length=self.config['sequence_length'],
            prediction_horizon=self.config['prediction_horizon'],
            features=self.config['features'],
            scaler_type=self.config['scaler_type']
        )
        processed_data = self.preprocessor.preprocess_data(
            data=raw_data,
            target_column='Close'
        )
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        print("Preprocessed data:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        print(f"   Features: {X_train.shape[2]}")
        
        # Step 3: Create and Train Model
        print("\nStep 3: Training Transformer Model")
        
        # Create model
        model = self.trainer.create_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            config=self.config['model_config']
        )
        
        # Compile model
        self.trainer.compile_model(model, learning_rate=self.config['learning_rate'])
        
        # Train model
        training_results = self.trainer.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            model_save_path=self.config['model_save_path']
        )
        
        print(f"Training completed in {training_results['training_time']:.2f}s")
        print(f"Best validation loss: {training_results['final_val_loss']:.6f}")
        
        # Step 4: Evaluate Model
        print("\nStep 4: Model Evaluation")
        
        # Load best model
        best_model = self.trainer.load_model(self.config['model_save_path'])
        
        # Get predictions
        y_pred = self.trainer.predict(best_model, X_test)
        
        # Calculate metrics
        metrics = self.trainer.evaluate(best_model, X_test, y_test)
        
        # Additional financial metrics
        financial_metrics = self.metrics.calculate_all_metrics(
            y_test.flatten(), 
            y_pred.flatten(), 
            "Informer"
        )
        
        print("\nFinal Results:")
        print(f"   MAE:  {metrics['mae']:.6f}")
        print(f"   RMSE: {metrics['rmse']:.6f}")
        print(f"   MAPE: {financial_metrics['MAPE']:.2f}%")
        print(f"   R²:   {financial_metrics['R2_Score']:.6f}")
        
        # Step 5: Generate Visualizations
        print("\nStep 5: Creating Visualizations")
        self._create_visualizations(y_test, y_pred, training_results['history'])
        
        # Step 6: Save Results
        self._save_results(metrics, financial_metrics, training_results)
        
        print("\nPipeline completed successfully!")
        print("Results saved in: results/")
        print(f"Model saved in: {self.config['model_save_path']}")
        
        return {
            'model': best_model,
            'metrics': {**metrics, **financial_metrics},
            'training_results': training_results,
            'predictions': y_pred,
            'preprocessor': self.preprocessor
        }
    
    def _create_visualizations(self, y_test, y_pred, history):
        """Create and save visualizations"""
        
        # 1. Training History
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Training History - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['mae'], label='Training MAE')
        plt.plot(history['val_mae'], label='Validation MAE')
        plt.title('Model Training History - MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Predictions vs Actual
        self.metrics.plot_predictions_vs_actual(
            y_test.flatten(),
            y_pred.flatten(),
            model_name=f"{self.config['symbol']} Transformer",
            save_path='plots/predictions_vs_actual.png'
        )
        
        print("Visualizations saved to plots/")
    
    def _save_results(self, metrics, financial_metrics, training_results):
        """Save results to files"""
        import json
        
        results = {
            'config': self.config,
            'metrics': {**metrics, **financial_metrics},
            'training_time': training_results['training_time'],
            'epochs_trained': training_results['epochs_trained'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open('results/experiment_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate text report
        report = f"""
Stock Price Forecasting Results
==============================

Symbol: {self.config['symbol']}
Date Range: {self.config['start_date']} to {self.config['end_date']}
Model: Informer Transformer

Configuration:
- Sequence Length: {self.config['sequence_length']} days
- Prediction Horizon: {self.config['prediction_horizon']} day(s)
- Model Parameters: {sum(p.numel() for p in []) if hasattr(self, 'model') else 'N/A'}

Performance Metrics:
- MAE:  {metrics['mae']:.6f}
- RMSE: {metrics['rmse']:.6f}
- MAPE: {financial_metrics['MAPE']:.2f}%
- R²:   {financial_metrics['R2_Score']:.6f}

Training:
- Epochs: {training_results['epochs_trained']}
- Training Time: {training_results['training_time']:.2f} seconds
- Final Validation Loss: {training_results['final_val_loss']:.6f}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        with open('results/report.txt', 'w') as f:
            f.write(report)
        
        print("Results saved to results/")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Stock Price Forecasting with Informer')
    
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Stock symbol to predict (default: AAPL)')
    parser.add_argument('--sequence-length', type=int, default=60,
                       help='Input sequence length in days (default: 60)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = {
        'symbol': args.symbol,
        'sequence_length': args.sequence_length,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }
    
    # Run pipeline
    pipeline = StockForecastingPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    return results

if __name__ == "__main__":
    results = main() 