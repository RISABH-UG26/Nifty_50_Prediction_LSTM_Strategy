0#  NIFTY 50 Price Prediction: 4 LSTM Approaches Comparison

[![Python 3.13](https://img.shields.io/badge/Python-3.13.5-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](#)

> A comprehensive deep learning project comparing **4 LSTM architectures** for NIFTY 50 price prediction. Demonstrates the power of integrating news sentiment with technical and macro-economic factors. **Novel approach achieves superior prediction accuracy.**

---

##  Quick Navigation
- [Project Overview](#project-overview)
- [4 LSTM Approaches](#4-lstm-approaches)
- [Quick Start](#quick-start)
- [Visualizations](#visualizations)
- [Technical Details](#technical-details)

---

##  Project Overview

This project systematically compares **4 LSTM architectures** for predicting NIFTY 50 closing prices:

| Approach | Features | Best For | Status |
|----------|----------|----------|--------|
| 1 Univariate LSTM | Close price only | Baseline comparison |  |
| 2 Multivariate LSTM | OHLCV (5) | Industry standard |  |
| 3 LSTM + External | OHLCV + Macro (8) | Global factors |  |
| 4 LSTM + Sentiment  | OHLCV + Macro + News (9) | **Best Performance** |  **NOVEL** |

**Key Innovation**: Approach 4 integrates news sentiment to capture market psychology, showing measurably superior prediction accuracy.

---

##  4 LSTM Approaches

### Approach 1: Univariate LSTM (Baseline)
- **Features**: [Close_Price]
- **Purpose**: Simple baseline
- **Model**: 2LSTM(64) + Dropout(0.2)

### Approach 2: Multivariate LSTM (Industry Standard)
- **Features**: [Open, High, Low, Close, Volume]
- **Purpose**: OHLCV best practice
- **Model**: 2LSTM(64) + Dropout(0.2)

### Approach 3: LSTM + External Factors
- **Features**: [OHLCV] + [VIX, S&P500_Return, USDINR_Return]
- **Purpose**: Global market correlation
- **Model**: 2LSTM(64) + Dropout(0.2)

### Approach 4: LSTM + Sentiment (NOVEL) 
- **Features**: [OHLCV] + [VIX, S&P500_Return, USDINR_Return, News_Sentiment]
- **Purpose**: Complete market psychology + technicals
- **Model**: 2LSTM(64) + Dropout(0.2)
- **Backtesting**: Trading signals vs Buy & Hold

**Why Novel**: Sentiment captures investor psychology not in traditional metrics. Backtesting shows measurable edge.

---

##  Quick Start

### Installation
```bash
# Clone repo
git clone https://github.com/RISABH-UG26/Nifty_Predict.git
cd Nifty_Predict

# Setup environment
python -m venv .venv
.\.venv\Scripts\activate          # Windows
source .venv/bin/activate         # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Step 1: Generate Dataset**
```bash
python data_collection.py
# Output: final_nifty_volatility_dataset.csv (2,328 rows  15 features)
```

**Step 2: Generate Signals (Optional)**
```bash
python algorithmic_trading.py
# Output: Sentiment scores and trading recommendations
```

**Step 3: Compare LSTM Models**
```bash
python forecast_future.py
# Output:
#   - nifty_lstm_comparison_predictions.png
#   - nifty_lstm_metrics_comparison.png
#   - nifty_strategy_performance.png
```

---

##  Visualizations

### 1. Model Comparison (22 Grid)
Shows predictions from all 4 models on first 100 test samples:
- **Top-left**: Univariate (Red)
- **Top-right**: Multivariate (Blue)
- **Bottom-left**: External Factors (Green)
- **Bottom-right**: Sentiment (Orange)  Best

### 2. Metrics Comparison (3 Bar Charts)
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **R**: Coefficient of Determination (higher is better)

### 3. Strategy Performance
- **Blue line**: Actual NIFTY 50 prices
- **Orange line**: Sentiment model predictions
- **Title**: Efficiency %, Strategy Return %, Buy&Hold Return %

---

##  Project Structure

```
Nifty_Predict/
 README.md                              # Documentation
 requirements.txt                       # Dependencies
 data_collection.py                     # Data aggregation
 algorithmic_trading.py                 # Sentiment analysis
 forecast_future.py                     # LSTM comparison
 final_nifty_volatility_dataset.csv    # Dataset
 nifty_lstm_comparison_predictions.png  # Visualization 1
 nifty_lstm_metrics_comparison.png      # Visualization 2
 nifty_strategy_performance.png         # Visualization 3
```

---

##  Technical Details

### Architecture
```
Input (60 days  N features)
    
LSTM Layer 1 (64 units)
    
Dropout (20%)
    
LSTM Layer 2 (64 units)
    
Dropout (20%)
    
Dense Output (1 unit - price prediction)
```

### Configuration
- **Dataset**: 2,328 trading days (Feb 2015 - Nov 2024)
- **Features**: 15 (OHLCV, VIX, S&P500, Oil, USD-INR, Sentiment)
- **Lookback**: 60 trading days
- **Train-Test**: 80/20 split (chronological)
- **Optimizer**: Adam
- **Loss**: Mean Squared Error (MSE)
- **Epochs**: 5
- **Batch Size**: 8

---

##  Key Findings

 **Approach 4 (Sentiment)** shows lowest MAE  
 **Approach 3** validates importance of macro factors  
 **Clear progression**: Simple  Complex = Better accuracy  
 **Backtesting**: Sentiment strategy outperforms Buy & Hold  

---

##  References

1. Hochreiter, S., & Schmidhuber, J. (1997) - Long Short-Term Memory
2. Graves, A., & Schmidhuber, J. (2005) - LSTM Applications
3. Pang, B., & Lee, L. (2008) - Opinion Mining and Sentiment Analysis
4. Fama, E. F., & French, K. R. (2015) - Multi-Factor Asset Pricing

---

##  Disclaimer

**For educational & research purposes only**
- Past performance  Future results
- Market prediction is uncertain
- Use at your own risk
- Not financial advice

---

##  Contributing

Contributions welcome! Areas for enhancement:
- Hyperparameter optimization
- Ensemble methods
- Real-time sentiment integration
- Multi-step forecasting
- Transaction cost modeling

---

##  Contact

**Author**: Risabh  
**GitHub**: [@RISABH-UG26](https://github.com/RISABH-UG26)  
**Project**: [Nifty_Predict](https://github.com/RISABH-UG26/Nifty_Predict)

---

##  License

MIT License - Free for educational and research use

---

**Status**:  Complete  
**Version**: 2.0 - 4 Approaches Comparison  
**Python**: 3.13.5 | **TensorFlow**: 2.x

 **If this helps, please star the repo!**
