# Stock Price Forecasting with Transformers

A comprehensive implementation of a Transformer model for stock price prediction using time series forecasting techniques.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Data Features](#data-features)
- [References & Resources](#references--resources)
- [Quick Commands Reference](#quick-commands-reference)

## Overview

This project implements a Transformer neural network for predicting stock prices using historical market data. The implementation is based on the "Attention Is All You Need" paper and adapted specifically for time series forecasting in financial markets.

**Key Features**:
- Transformer model with multi-head attention for time series
- Automatic stock data fetching with fallback mechanisms
- Comprehensive data preprocessing with technical indicators
- Multiple evaluation metrics (MAE, RMSE, MAPE, R²)
- Modular, extensible architecture
- Complete visualization pipeline

**Project Goals**:
- Predict next-day stock closing prices
- Demonstrate Transformer effectiveness in financial time series
- Provide a foundation for further research and development
- Implement industry-standard evaluation practices

## Getting Started

### For Complete Beginners

If you're new to machine learning or this project, follow these steps:

1. **Understand the Goal**: We're building a system that looks at historical stock prices and tries to predict future prices using artificial intelligence.

2. **Key Concepts**:
   - **Time Series**: Data points ordered by time (daily stock prices)
   - **Transformer**: A type of neural network originally designed for language translation
   - **Attention Mechanism**: Allows the model to focus on important parts of the data
   - **Features**: Input variables (price, volume, technical indicators)
   - **Prediction**: The model's guess about future stock prices

3. **What the System Does**:
   - Downloads historical stock data (or generates realistic fake data for testing)
   - Calculates technical indicators (moving averages, RSI, etc.)
   - Trains a Transformer model to recognize patterns
   - Makes predictions on new data
   - Evaluates how accurate the predictions are

4. **Start Here**:
   ```bash
   # First, install dependencies
   pip install -r requirements.txt
   
   # Run a quick demo to see data processing (30 seconds)
   python demo.py --data-only
   
   # Run a quick training demo (2-3 minutes)
   python demo.py
   
   # Run the full pipeline (10-15 minutes)
   python main.py
   ```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone or download this project**
   ```bash
   # If you have git:
   git clone [repository-url]
   cd stock-forecasting
   
   # Or extract the ZIP file and navigate to the folder
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python demo.py --data-only
   ```

### Troubleshooting Installation

- **If you get "python not found"**: Try `python3` instead of `python`
- **If pip fails**: Try `pip3` instead of `pip`
- **On macOS**: You might need to install Xcode command line tools: `xcode-select --install`
- **Permission errors**: Try adding `--user` flag: `pip install --user -r requirements.txt`

## Project Structure

```
stock-forecasting/
├── main.py                    # Main pipeline - start here
├── demo.py                    # Quick demo scripts
├── requirements.txt           # Python dependencies
├── README.md                  # This documentation
├── project_proposal.md        # Original project requirements
│
├── src/                       # Source code
│   ├── data/                  # Data handling modules
│   │   ├── data_fetcher.py         # Stock data fetching
│   │   └── stock_preprocessor.py   # Data preprocessing
│   │
│   ├── models/                # Model implementations
│   │   ├── transformer.py          # Transformer model
│   │   └── trainer.py              # Training pipeline
│   │
│   └── evaluation/            # Evaluation and metrics
│       └── metrics.py              # Performance metrics
│
├── data/                      # Data directories (created when needed)
│   ├── raw/                        # Raw downloaded data
│   ├── processed/                  # Preprocessed data
│   └── external/                   # External datasets
│
├── models/                    # Saved models (created when training)
├── results/                   # Training results (created when training)
├── plots/                     # Generated visualizations (created when training)
└── notebooks/                 # Jupyter notebooks
    └── data_exploration.ipynb      # Data analysis notebook
```

### Key Files Explained

- **`main.py`**: The complete pipeline from data fetching to model evaluation
- **`demo.py`**: Quick demonstrations for learning and testing
- **`src/models/transformer.py`**: The core Transformer model implementation
- **`src/models/trainer.py`**: Handles model training, saving, and loading
- **`src/data/data_fetcher.py`**: Downloads stock data with multiple fallback sources
- **`src/data/stock_preprocessor.py`**: Converts raw data into model-ready format
- **`src/evaluation/metrics.py`**: Calculates all performance metrics

## Data Pipeline

### Data Sources

The system uses multiple data sources with automatic fallback:

1. **Yahoo Finance (Primary)**: Free, reliable historical data
2. **Alternative APIs**: Backup sources for data fetching
3. **Mock Data Generator**: Creates realistic synthetic data for development

### Data Processing Steps

1. **Fetching**: Download OHLCV (Open, High, Low, Close, Volume) data
2. **Technical Indicators**: Calculate moving averages, RSI, MACD, etc.
3. **Cleaning**: Handle missing values and outliers
4. **Scaling**: Normalize data for neural network training
5. **Sequencing**: Create sliding windows of historical data
6. **Splitting**: Divide into training, validation, and test sets

### Technical Indicators Added

- **SMA/EMA**: Simple and Exponential Moving Averages
- **MACD**: Moving Average Convergence Divergence
- **RSI**: Relative Strength Index
- **Bollinger Bands**: Price volatility bands
- **Price Returns**: Daily percentage changes
- **Volatility**: Rolling standard deviation of returns

## Model Architecture

### Transformer Architecture Overview

Our Transformer model is based on the "Attention Is All You Need" paper, adapted for time series forecasting:

```
Input: [Batch, Sequence_Length, Features]
   ↓
Input Projection: Linear layer to model dimension
   ↓
Positional Encoding: Add position information
   ↓
Transformer Blocks (Multiple layers):
┌─────────────────────────────────────┐
│  Multi-Head Self-Attention         │
│           ↓                        │
│  Add & Layer Normalization         │
│           ↓                        │
│  Feed-Forward Network              │
│           ↓                        │
│  Add & Layer Normalization         │
└─────────────────────────────────────┘
   ↓
Global Average Pooling: Aggregate sequence information
   ↓
Dense Layer: Final processing
   ↓
Output: Single predicted price
```

### Architecture Components

1. **Multi-Head Attention**:
   - Allows the model to focus on different parts of the input sequence
   - Multiple attention heads capture different types of relationships
   - Queries, Keys, and Values are learned transformations of the input

2. **Positional Encoding**:
   - Adds information about the position of each day in the sequence
   - Uses sinusoidal functions to encode temporal relationships
   - Crucial for time series as it maintains order information

3. **Feed-Forward Networks**:
   - Two linear layers with ReLU activation
   - Processes the attention output for each position independently
   - Adds non-linear transformations to the model

4. **Layer Normalization**:
   - Stabilizes training by normalizing inputs to each layer
   - Applied after attention and feed-forward operations

5. **Residual Connections**:
   - Skip connections that help with gradient flow
   - Allow the model to learn identity mappings when needed

### Model Configuration

**Default Parameters**:
```python
{
    'sequence_length': 60,    # 60 days of historical data
    'prediction_horizon': 1,  # Predict next day
    'd_model': 128,          # Model dimension
    'num_heads': 8,          # Number of attention heads
    'num_layers': 4,         # Number of transformer blocks
    'dff': 512,              # Feed-forward dimension
    'dropout_rate': 0.1      # Dropout for regularization
}
```

### Why Transformers for Time Series?

1. **Long-Range Dependencies**: Can connect events far apart in time
2. **Parallel Processing**: Faster training compared to RNNs
3. **Attention Interpretation**: Can visualize what the model focuses on
4. **Scalability**: Performance improves with more data and parameters
5. **Flexibility**: Easy to modify for different prediction horizons

## Usage

### Basic Usage

1. **Quick Demo** (recommended for first-time users):
   ```bash
   python demo.py --data-only  # See data processing only
   python demo.py             # Quick training demo
   ```

2. **Full Pipeline**:
   ```bash
   python main.py  # Complete training and evaluation
   ```

3. **Custom Stock**:
   ```bash
   python main.py --symbol TSLA  # Train on Tesla stock
   ```

### Advanced Usage

**Custom Configuration**:
```bash
python main.py --symbol GOOGL --epochs 100 --sequence-length 30
```

**Available Command Line Options**:
- `--symbol`: Stock symbol (default: AAPL)
- `--sequence-length`: Days of history to use (default: 60)
- `--epochs`: Training epochs (default: 100)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Learning rate (default: 1e-4)

### Understanding the Output

When you run the pipeline, you'll see:

1. **Data Fetching**: Progress of downloading stock data
2. **Preprocessing**: Feature engineering and data preparation
3. **Model Training**: Training progress with loss values
4. **Evaluation**: Performance metrics on test data
5. **Visualization**: Plots saved to the `plots/` directory
6. **Results**: Detailed results saved to `results/` directory

### Interpreting Results

**Training Progress**:
- **Loss**: Should decrease over epochs (lower is better)
- **Validation Loss**: Should track training loss (if much higher, overfitting)
- **MAE**: Mean Absolute Error in dollars (lower is better)

**Final Metrics**:
- **MAE**: Average prediction error in dollars
- **RMSE**: Root Mean Square Error (penalizes large errors)
- **MAPE**: Mean Absolute Percentage Error (scale-independent)
- **R²**: Coefficient of determination (closer to 1 is better)

## Configuration

### Configuration System

The project uses a simple dictionary-based configuration system that can be modified in `main.py` or passed via command line arguments.

### Default Configuration

```python
{
    # Data configuration
    'symbol': 'AAPL',               # Stock ticker symbol
    'start_date': '2019-01-01',     # Start of data range
    'end_date': '2024-01-01',       # End of data range
    
    # Preprocessing
    'sequence_length': 60,          # Days of history
    'prediction_horizon': 1,        # Days to predict ahead
    'scaler_type': 'standard',      # 'standard' or 'minmax'
    'features': ['Open', 'High', 'Low', 'Close', 'Volume'],
    
    # Model configuration
    'model_config': {
        'd_model': 128,             # Model dimension
        'num_heads': 8,             # Attention heads
        'num_layers': 4,            # Transformer layers
        'dff': 512,                 # Feed-forward dimension
        'dropout_rate': 0.1         # Dropout rate
    },
    
    # Training
    'epochs': 100,                  # Training epochs
    'batch_size': 32,               # Batch size
    'learning_rate': 1e-4,          # Learning rate
    'patience': 15,                 # Early stopping patience
    
    # Output
    'model_save_path': 'models/transformer_stock_model'
}
```

### Modifying Configuration

**Method 1: Edit main.py**
```python
# In main.py, modify the _get_default_config method
config['epochs'] = 200              # Train longer
config['sequence_length'] = 30      # Use 30 days instead of 60
config['model_config']['d_model'] = 256  # Larger model
```

**Method 2: Command Line**
```bash
python main.py --epochs 200 --sequence-length 30
```

## Evaluation Metrics

### Core Metrics

1. **MAE (Mean Absolute Error)**:
   - Average absolute difference between predicted and actual prices
   - Measured in dollars
   - Lower values indicate better performance
   - Easy to interpret: "On average, predictions are off by $X"

2. **RMSE (Root Mean Square Error)**:
   - Square root of average squared errors
   - Penalizes large errors more heavily than MAE
   - Measured in dollars
   - More sensitive to outliers

3. **MAPE (Mean Absolute Percentage Error)**:
   - Average absolute percentage difference
   - Scale-independent (useful for comparing different stocks)
   - Expressed as percentage
   - Can be problematic when actual values are near zero

4. **R² Score (Coefficient of Determination)**:
   - Proportion of variance explained by the model
   - Ranges from -∞ to 1
   - 1 = perfect predictions, 0 = no better than average, negative = worse than average

### Additional Metrics

- **Directional Accuracy**: Percentage of correct up/down predictions
- **Max Error**: Largest single prediction error
- **Trading Simulation**: Profit/loss from simulated trading based on predictions

### Interpreting Performance

**Good Performance Indicators**:
- MAPE < 10% (excellent), < 20% (good)
- R² > 0.5 (reasonable), > 0.8 (good)
- Directional accuracy > 55% (better than random)

**Warning Signs**:
- Training loss much lower than validation loss (overfitting)
- Very high R² (> 0.99) might indicate data leakage
- MAPE > 50% suggests poor model performance

## Data Features

### Raw Features (OHLCV)

- **Open**: Opening price of the trading day
- **High**: Highest price during the trading day
- **Low**: Lowest price during the trading day
- **Close**: Closing price of the trading day
- **Volume**: Number of shares traded

### Technical Indicators (Automatically Generated)

1. **Moving Averages**:
   - SMA (Simple Moving Average): Average price over N days
   - EMA (Exponential Moving Average): Weighted average favoring recent prices

2. **Momentum Indicators**:
   - RSI (Relative Strength Index): Measures overbought/oversold conditions
   - MACD (Moving Average Convergence Divergence): Trend-following momentum

3. **Volatility Indicators**:
   - Bollinger Bands: Price channels based on standard deviation
   - Volatility: Rolling standard deviation of returns

4. **Price Indicators**:
   - Returns: Daily percentage price changes
   - Price ratios: Current price relative to moving averages

### Feature Selection

The preprocessing pipeline automatically:
- Calculates all technical indicators
- Handles missing values through forward-fill and interpolation
- Scales features to have similar ranges
- Selects the most informative features for model input

## References & Resources

### Key Papers

- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Temporal Fusion Transformers (2021)](https://arxiv.org/abs/1912.09363) - Transformers for time series
- [Informer (2021)](https://arxiv.org/abs/2012.07436) - Efficient long sequence transformers

### Learning Resources

- [Transformer Tutorial](https://www.tensorflow.org/text/tutorials/transformer) - Official TensorFlow guide
- [Time Series with Deep Learning](https://www.tensorflow.org/tutorials/structured_data/time_series) - TensorFlow time series tutorial
- [Understanding Transformers](https://jalammar.github.io/illustrated-transformer/) - Visual explanation

### Technical Documentation

- [TensorFlow/Keras Documentation](https://tensorflow.org/api_docs)
- [Pandas Documentation](https://pandas.pydata.org/docs/) - Data manipulation
- [NumPy Documentation](https://numpy.org/doc/) - Numerical computing

## Quick Commands Reference

```bash
# Essential commands for getting started
python demo.py --data-only        # Understand data flow (30 seconds)
python demo.py                    # See training in action (2-3 minutes)
python main.py                    # Full pipeline (10-15 minutes)

# Experiment with different stocks
python main.py --symbol TSLA      # Tesla
python main.py --symbol GOOGL     # Google
python main.py --symbol MSFT      # Microsoft

# Adjust training parameters
python main.py --epochs 50        # Faster training
python main.py --epochs 200       # More thorough training
python main.py --batch-size 16    # Smaller batches (less memory)
python main.py --sequence-length 30  # Shorter sequences (faster)

# Help and information
python main.py --help             # See all available options
python demo.py --help             # Demo script options
```
