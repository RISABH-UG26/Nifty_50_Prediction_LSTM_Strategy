# NIFTY 50 Price Prediction using Ensemble LSTM-RNN

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MAE](https://img.shields.io/badge/MAE-149.14%20INR-brightgreen.svg)](#results)
[![R²](https://img.shields.io/badge/R²-0.9036-brightgreen.svg)](#results)

> **High-accuracy stock price prediction** using an ensemble of BiLSTM, BiGRU, and CNN-LSTM models. Achieves **MAE of 149 INR** and **R² of 0.90** on NIFTY 50 index prediction.

---

## 🎯 Results at a Glance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **MAE** | 149.14 INR | < 300 INR | ✅ Achieved |
| **RMSE** | 202.18 INR | - | - |
| **R²** | 0.9036 | > 0.85 | ✅ Achieved |
| **MAPE** | 0.61% | - | Excellent |

---

## 🔑 Key Innovation

The critical breakthrough was **predicting price CHANGES instead of absolute prices**:

```
❌ BEFORE: Predict tomorrow's price directly → MAE ~4000 INR (poor)
✅ AFTER:  Predict price CHANGE, add to current price → MAE ~150 INR (excellent)
```

This works because price changes are more stationary and easier for neural networks to learn.

---

## 🏗️ Architecture

### Ensemble of 3 Models

```
Input (20 days × 33 features)
         │
    ┌────┴────┬────────────┐
    ▼         ▼            ▼
 BiLSTM    BiGRU      CNN-LSTM
    │         │            │
    ▼         ▼            ▼
  Pred1     Pred2       Pred3
    │         │            │
    └────┬────┴────────────┘
         ▼
  Weighted Average (based on validation MAE)
         │
         ▼
  Predicted Price Change
         │
         ▼
  Current Price + Change = Predicted Price
```

### Individual Model Performance

| Model | MAE (INR) | R² | Weight |
|-------|-----------|-----|--------|
| BiLSTM | 151.71 | 0.9028 | 33.3% |
| BiGRU | 149.18 | 0.9033 | 33.5% |
| CNN-LSTM | 148.37 | 0.9037 | 33.3% |
| **Ensemble** | **149.14** | **0.9036** | 100% |

---

## 📊 Features Used (33 total)

| Category | Features | Purpose |
|----------|----------|---------|
| **Price** | Close, Open, High, Low | Raw price data |
| **Lagged** | Close_Lag1-5, Return_Lag1-3 | Historical patterns |
| **Moving Avg** | SMA5/10/20, Price ratios | Trend detection |
| **Momentum** | RSI, Stochastic, MACD, ROC | Overbought/oversold |
| **Volatility** | Bollinger Bands, ATR, HV | Risk measurement |
| **External** | VIX, Volume Ratio | Market sentiment |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Trade_algo.git
cd Trade_algo

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Model

```bash
python nifty_lstm_predictor.py
```

**Expected Output:**
```
================================================================================
NIFTY 50 FINAL OPTIMIZED LSTM-RNN PREDICTOR
================================================================================

📊 Loading and preparing data...
🔬 Engineering features...
⚙️ Preparing sequences...
🚀 Training ensemble...
   Training BiLSTM... ✓
   Training BiGRU... ✓
   Training CNN-LSTM... ✓

📊 FINAL RESULTS
================================================================================
🏆 ENSEMBLE PERFORMANCE:
   MAE:  149.14 INR
   RMSE: 202.18 INR
   R²:   0.9036
   MAPE: 0.61%
```

### 3. Update Data (Optional)

```bash
python data_collection.py
```

---

## 📁 Project Structure

```
Trade_algo/
├── nifty_lstm_predictor.py      # Main ensemble LSTM model
├── data_collection.py           # Data fetching script
├── final_nifty_volatility_dataset.csv  # Historical NIFTY data
├── nifty_lstm_final_results.png # Prediction visualization
├── nifty_model_comparison.png   # Model comparison chart
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── RESEARCH_PAPER.txt           # Detailed documentation
```

---

## 📈 Visualizations

The model generates two visualization files:

### 1. Prediction Results (`nifty_lstm_final_results.png`)
- Actual vs Predicted prices
- Last 30 days zoomed view
- Error distribution histogram
- Scatter plot of predictions

### 2. Model Comparison (`nifty_model_comparison.png`)
- MAE comparison bar chart
- Target line at 300 INR
- Individual vs Ensemble performance

---

## ⚙️ Technical Details

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Lookback Window | 20 days |
| Train/Val/Test Split | 80/10/10 |
| Optimizer | Adam (lr=0.001) |
| Loss Function | MSE |
| Batch Size | 32 |
| Max Epochs | 150 |
| Early Stopping Patience | 25 |
| Scaling | RobustScaler |

### Model Architectures

**BiLSTM:**
```
BiLSTM(64) → Dropout(0.2) → BiLSTM(64) → Dropout(0.2) → 
BiLSTM(32) → Dropout(0.2) → Dense(64) → BatchNorm → 
Dense(32) → Dense(16) → Dense(1)
```

**BiGRU:**
```
BiGRU(64) → Dropout(0.2) → BiGRU(32) → Dropout(0.2) → 
Dense(64) → BatchNorm → Dense(32) → Dense(1)
```

**CNN-LSTM:**
```
Conv1D(64) → Conv1D(64) → MaxPool → Dropout(0.2) → 
LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(64) → Dense(32) → Dense(1)
```

---

## 📚 Requirements

```
tensorflow>=2.10
pandas
numpy
scikit-learn
matplotlib
yfinance
requests
beautifulsoup4
nltk
```

---

## 🔮 Future Improvements

1. **Direction Prediction**: Add classification head for up/down prediction
2. **Uncertainty Estimation**: Monte Carlo Dropout for confidence intervals
3. **Real-time Integration**: Live news sentiment and intraday data
4. **Hyperparameter Tuning**: Optuna/Ray Tune optimization

---

## 📄 Documentation

For detailed technical documentation, see [RESEARCH_PAPER.txt](RESEARCH_PAPER.txt) which includes:
- Complete feature engineering explanation
- Model architecture details
- Training process documentation
- Results analysis and insights

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Yahoo Finance API for market data
- TensorFlow/Keras for deep learning framework
- NIFTY 50 historical data providers

---

**Made with ❤️ by Risabh**
