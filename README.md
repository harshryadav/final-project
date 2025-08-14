# Stock Price Forecasting with Transformer & Temporal Fusion Transformer (TFT)

This project implements **two state-of-the-art deep learning architectures** for short-term stock price forecasting:
1. **Transformer** (TensorFlow/Keras)
2. **Temporal Fusion Transformer (TFT)** (PyTorch)

It supports multiple stock symbols, customizable hyperparameters, automatic technical indicator generation, and detailed evaluation metrics.

---

## Features

- **Dual-Model Support** → Transformer (fast & accurate) and TFT (interpretable, multi-horizon forecasting)
- **Configurable Parameters** → Sequence length, features, batch size, learning rate, patience
- **Automatic Mixed Precision (AMP)** → Faster GPU training with reduced memory usage
- **Dynamic Close Index Detection** → No hardcoded target index
- **Rich Feature Engineering** → SMA, EMA, MACD, RSI, Bollinger Bands
- **Complete Evaluation** → MAE, RMSE, MAPE, R², Directional Accuracy, Sharpe Ratio, Total Return
- **Live Demo Script** (`demo.py`) → Quick training/inference without full training

---

## Installation

### Prerequisites
- Python **3.8+**
- pip

### Setup
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd final-project-main

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### **Train & Evaluate**

#### Transformer (AAPL example)
```bash
python main.py --model transformer --symbol AAPL --sequence-length 60 --batch-size 256 --epochs 50 --learning-rate 1e-4 --patience 7
```

#### TFT (NVDA example)
```bash
python main.py --model tft --symbol NVDA --sequence-length 60 --batch-size 256 --epochs 50 --learning-rate 3e-4 --patience 7
```

**Key CLI Arguments**
| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Model type (`transformer` / `tft`) | `--model transformer` |
| `--symbol` | Stock ticker | `--symbol AAPL` |
| `--sequence-length` | Lookback window size | `--sequence-length 60` |
| `--prediction-horizon` | Days ahead to predict | `--prediction-horizon 1` |
| `--epochs` | Training epochs | `--epochs 50` |
| `--batch-size` | Training batch size | `--batch-size 256` |
| `--learning-rate` | Learning rate | `--learning-rate 1e-4` |
| `--patience` | Early stopping patience | `--patience 7` |

---

### **Quick Demo**
Run a short demo with reduced parameters:
```bash
python demo.py --model transformer --symbol AAPL
python demo.py --model tft --symbol NVDA
```

Run only data processing (no training):
```bash
python demo.py --data-only --symbol TSLA
```

---

## Final Results (Interim → Final)

| Model       | Symbol | R² Score (Interim) | R² Score (Final) | MAE Final | RMSE Final |
|-------------|--------|-------------------|------------------|-----------|------------|
| Transformer | AAPL   | -0.4468           | **0.9066**       | 0.079     | 0.101      | 
| TFT         | NVDA   | 0.0707            | 0.1257           | 0.602     | 0.754      | 

Transformer → **Major accuracy improvement** after efficiency optimizations
TFT → Minor gains, requires further hyperparameter tuning.

---

## Project Structure

```
final-project-main/
├── main.py               # Main training & evaluation pipeline
├── trainer.py            # PyTorch training loop for TFT
├── transformer.py        # Transformer model
├── tft.py                # Temporal Fusion Transformer model
├── stock_preprocessor.py # Data preprocessing & feature engineering
├── demo.py               # Quick demo script
├── requirements.txt      # Python dependencies
├── results/              # Metrics and reports
├── plots/                # Generated plots
└── models/               # Saved model checkpoints
```

---

## Model Architectures

**Transformer**
```
[Input Sequence] → Input Projection → Positional Encoding
→ Multi-Head Attention → Layer Norm → Feed-Forward → Layer Norm
→ Global Average Pooling → Dense → Output
```

**TFT**
```
Static Variable Encoder + Historical Encoder
→ Multi-Head Attention + LSTM Layers
→ Temporal Decoder → Output Layer
```

Below is a detailed, end-to-end breakdown of the data and model architecture for both the Transformer and TFT implementations.

### End-to-End Workflow

Both models follow the same initial data processing pipeline:

1.  **Data Fetching**:
    *   **Input**: Stock Symbol (e.g., `AAPL`), Start Date, End Date.
    *   **Action**: Use the `yfinance` library to download historical daily data (Open, High, Low, Close, Volume).
    *   **Output**: A raw `pandas.DataFrame`.

2.  **Feature Engineering & Preprocessing**:
    *   **Input**: The raw DataFrame.
    *   **Action**:
        *   Calculate a rich set of technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, Volatility).
        *   Select the features to be used in the model (e.g., `Close`, `Volume`, `RSI`).
        *   Scale all selected features to a common range using `StandardScaler` or `MinMaxScaler`.
    *   **Output**: A scaled DataFrame of engineered features.

3.  **Sequence Creation**:
    *   **Input**: The scaled feature DataFrame.
    *   **Action**: Create overlapping time-series sequences. For each sample, the input (`X`) is a window of the last `N` days (e.g., 60 days), and the target (`y`) is the 'Close' price of the following day.
    *   **Output**: `(X_train, y_train)`, `(X_val, y_val)`, and `(X_test, y_test)` NumPy arrays ready for training.

---

#### 1. Transformer (TensorFlow/Keras) Architecture

The Transformer model processes the sequences to capture temporal patterns and predict the *change* in the next day's stock price.

```
[Input Sequence (Batch, 60, 7)]
           │
           ▼
┌──────────────────────────┐
│  Input Projection (Dense)│  (Projects 7 features to 128)
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  Positional Encoding     │  (Adds time-step information)
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  Transformer Encoder x4  │
│  ┌─────────────────────┐ │
│  │ Multi-Head Attention│ │  (Learns relationships across the 60 days)
│  └─────────────────────┘ │
│  ┌─────────────────────┐ │
│  │ Feed-Forward Network│ │
│  └─────────────────────┘ │
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Global Average Pooling   │  (Condenses the sequence into one vector)
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  Output Head (Dense)     │  (Outputs a single value: the predicted price *delta*)
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  Final Price Calculation │  (Last Close Price + Predicted Delta)
└──────────────────────────┘
           │
           ▼
  [Predicted Stock Price]
```

---

#### 2. Temporal Fusion Transformer (TFT) (PyTorch) Architecture

The TFT model uses specialized components like variable selection networks and an LSTM to interpret the time-series data.

```
[Input Sequence (Batch, 60, 7)]
           │
           ▼
┌──────────────────────────┐
│ Variable Selection Net   │  (Learns which features are important at each time step)
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  LSTM Encoder            │  (Processes the weighted features sequentially)
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  Multi-Head Attention    │  (Focuses on the most relevant past time steps)
│  - Query: LSTM State     │
│  - Key/Value: LSTM Output│
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Gated Residual Network   │  (Post-attention processing)
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  Output Head (Linear)    │  (Outputs a single value: the predicted price)
└──────────────────────────┘
           │
           ▼
  [Predicted Stock Price]
```

---

## License
MIT License – free to use, modify, and distribute.

---

## Acknowledgements
- Yahoo Finance for data
- PyTorch, TensorFlow/Keras for model implementation
- Original TFT paper: *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*
