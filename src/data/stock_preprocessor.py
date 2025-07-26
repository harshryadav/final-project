"""
Stock Data Preprocessor for Transformer Models

This module handles preprocessing of stock data for time series forecasting,
including feature engineering, scaling, and sequence creation for both
Informer and LSTM models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class StockDataPreprocessor:
    """
    Comprehensive preprocessor for stock data to prepare for Transformer models
    """
    
    def __init__(self, 
                 sequence_length: int = 30,
                 prediction_horizon: int = 1,
                 features: List[str] = None,
                 scaler_type: str = 'standard'):
        """
        Initialize the preprocessor
        
        Args:
            sequence_length: Number of days to look back (30 as per proposal)
            prediction_horizon: Number of days to predict ahead (1-5 as per proposal)
            features: List of features to use
            scaler_type: 'standard' or 'minmax'
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler_type = scaler_type
        
        # Default features as mentioned in proposal
        self.features = features or ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Initialize scalers
        self.feature_scalers = {}
        self.target_scaler = None
        
        # Store processed data
        self.processed_data = None
        self.sequences = None
        self.targets = None
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators to enhance the feature set
        """
        data = df.copy()
        
        # Moving averages
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        
        # Exponential moving averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        data['BB_middle'] = data['Close'].rolling(window=bb_period).mean()
        bb_std = data['Close'].rolling(window=bb_period).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        data['BB_width'] = data['BB_upper'] - data['BB_lower']
        data['BB_position'] = (data['Close'] - data['BB_lower']) / data['BB_width']
        
        # Volatility
        data['Volatility'] = data['Close'].rolling(window=20).std()
        
        # Price returns
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Price position
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['Close_Open_Ratio'] = data['Close'] / data['Open']
        
        return data
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        """
        data = df.copy()
        
        # Forward fill first, then backward fill
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # If still NaN values, drop those rows
        if data.isnull().sum().sum() > 0:
            print(f"Warning: Dropping {data.isnull().sum().sum()} NaN values")
            data = data.dropna()
        
        return data
    
    def _create_scalers(self, data: pd.DataFrame, target_column: str = 'Close'):
        """
        Create and fit scalers for features and target
        """
        if self.scaler_type == 'standard':
            ScalerClass = StandardScaler
        else:
            ScalerClass = MinMaxScaler
        
        # Create scalers for each feature
        for column in data.columns:
            if column != target_column:
                self.feature_scalers[column] = ScalerClass()
                self.feature_scalers[column].fit(data[column].values.reshape(-1, 1))
        
        # Create scaler for target
        self.target_scaler = ScalerClass()
        self.target_scaler.fit(data[target_column].values.reshape(-1, 1))
    
    def _scale_data(self, data: pd.DataFrame, target_column: str = 'Close') -> pd.DataFrame:
        """
        Scale the data using fitted scalers
        """
        scaled_data = data.copy()
        
        # Scale features
        for column in data.columns:
            if column != target_column and column in self.feature_scalers:
                scaled_data[column] = self.feature_scalers[column].transform(
                    data[column].values.reshape(-1, 1)
                ).flatten()
        
        # Scale target
        scaled_data[target_column] = self.target_scaler.transform(
            data[target_column].values.reshape(-1, 1)
        ).flatten()
        
        return scaled_data
    
    def _create_sequences(self, data: pd.DataFrame, target_column: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Returns:
            X: Input sequences of shape (samples, sequence_length, features)
            y: Target values of shape (samples, prediction_horizon)
        """
        # Select features for X
        feature_columns = [col for col in data.columns if col != target_column]
        X_data = data[feature_columns].values
        y_data = data[target_column].values
        
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            X.append(X_data[i:(i + self.sequence_length)])
            
            # Target sequence
            if self.prediction_horizon == 1:
                y.append(y_data[i + self.sequence_length])
            else:
                y.append(y_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def preprocess_data(self, 
                       data: pd.DataFrame,
                       target_column: str = 'Close',
                       add_technical_indicators: bool = True,
                       test_size: float = 0.2,
                       validation_size: float = 0.1) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            data: Raw stock data
            target_column: Column to predict
            add_technical_indicators: Whether to add technical indicators
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
        
        Returns:
            Dictionary containing train/val/test splits
        """
        print("Starting data preprocessing...")
        
        # Step 1: Add technical indicators
        if add_technical_indicators:
            print("Adding technical indicators...")
            processed_data = self._create_technical_indicators(data)
        else:
            processed_data = data.copy()
        
        # Step 2: Handle missing values
        print("Handling missing values...")
        processed_data = self._handle_missing_values(processed_data)
        
        # Step 3: Select features
        if self.features:
            # Ensure target column is included
            all_features = list(set(self.features + [target_column]))
            # Filter to available columns
            available_features = [col for col in all_features if col in processed_data.columns]
            processed_data = processed_data[available_features]
            print(f"Selected features: {available_features}")
        
        # Step 4: Create and fit scalers
        print("Fitting scalers...")
        self._create_scalers(processed_data, target_column)
        
        # Step 5: Scale data
        print("Scaling data...")
        scaled_data = self._scale_data(processed_data, target_column)
        
        # Step 6: Create sequences
        print("Creating sequences...")
        X, y = self._create_sequences(scaled_data, target_column)
        
        # Step 7: Split data
        print("Splitting data...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Second split: separate train and validation
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, shuffle=False
        )
        
        # Store processed data
        self.processed_data = processed_data
        self.sequences = X
        self.targets = y
        
        result = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': [col for col in scaled_data.columns if col != target_column],
            'target_name': target_column,
            'scaler_info': {
                'feature_scalers': self.feature_scalers,
                'target_scaler': self.target_scaler
            }
        }
        
        print(f"Preprocessing complete!")
        print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
        print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
        
        return result
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Convert scaled predictions back to original scale
        """
        if self.target_scaler is None:
            raise ValueError("Target scaler not fitted. Run preprocess_data first.")
        
        # Reshape if necessary
        original_shape = predictions.shape
        if predictions.ndim > 1:
            predictions_reshaped = predictions.reshape(-1, 1)
        else:
            predictions_reshaped = predictions.reshape(-1, 1)
        
        # Inverse transform
        inverse_predictions = self.target_scaler.inverse_transform(predictions_reshaped)
        
        # Reshape back to original shape
        return inverse_predictions.reshape(original_shape)
    
    def get_feature_info(self) -> Dict:
        """
        Get information about features and preprocessing
        """
        if self.processed_data is None:
            return {"error": "No data processed yet"}
        
        info = {
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "total_features": len(self.features) if self.features else len(self.processed_data.columns) - 1,
            "feature_names": self.features or list(self.processed_data.columns),
            "data_shape": self.processed_data.shape,
            "scaler_type": self.scaler_type,
            "sequences_shape": self.sequences.shape if self.sequences is not None else None,
            "targets_shape": self.targets.shape if self.targets is not None else None
        }
        
        return info

# Convenience function for quick preprocessing
def preprocess_stock_data(data: pd.DataFrame, 
                         sequence_length: int = 30,
                         prediction_horizon: int = 1,
                         features: List[str] = None) -> Dict:
    """
    Quick preprocessing function for stock data
    """
    preprocessor = StockDataPreprocessor(
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        features=features
    )
    
    return preprocessor.preprocess_data(data)

if __name__ == "__main__":
    # Demo with sample data
    from data_fetcher import fetch_stock_data
    
    print("=== Stock Data Preprocessor Demo ===")
    
    # Fetch sample data
    data = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')
    
    if data is not None:
        # Create preprocessor
        preprocessor = StockDataPreprocessor(sequence_length=30, prediction_horizon=1)
        
        # Preprocess data
        result = preprocessor.preprocess_data(data)
        
        # Show results
        print("\nPreprocessing Results:")
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {type(value)}")
        
        # Feature info
        print("\nFeature Information:")
        info = preprocessor.get_feature_info()
        for key, value in info.items():
            print(f"{key}: {value}")
    else:
        print("Could not fetch data for demo") 