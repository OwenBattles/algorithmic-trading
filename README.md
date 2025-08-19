# Algorithmic Trading System

A comprehensive algorithmic trading system that uses machine learning models to predict stock price movements and execute automated trading strategies.

## Project Overview

This system implements a machine learning-based approach to algorithmic trading, featuring:
- Pre-trained ML models for multiple stocks (AAPL, AMZN, KO, MSFT)
- Automated trading strategies based on technical indicators
- Portfolio performance tracking and visualization
- Web-based trading dashboard
- Comprehensive backtesting capabilities

## Features

### Core Trading System
- **Multi-Stock Support**: Trade AAPL, AMZN, KO, and MSFT simultaneously
- **ML-Powered Predictions**: Uses Random Forest models trained on technical indicators
- **Risk Management**: Configurable thresholds for buy/sell decisions
- **Portfolio Tracking**: Real-time monitoring of holdings and performance

### Technical Indicators
- RSI (Relative Strength Index)
- Stochastic Oscillator (K% and R%)
- Price Rate of Change
- MACD (Moving Average Convergence Divergence)
- On Balance Volume

### Visualization & Analysis
- Portfolio performance charts
- Individual stock performance tracking
- Trading activity summaries
- Performance metrics and statistics

## Project Structure

```
Algorithmic Trading/
├── daily_trades.py              # Main trading simulation engine
├── graph.py                     # Portfolio visualization and analysis
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── stock_notebooks/            # Stock-specific analysis notebooks
│   ├── aapl.ipynb             # Apple stock analysis
│   ├── amzn.ipynb             # Amazon stock analysis
│   ├── ko.ipynb               # Coca-Cola stock analysis
│   ├── msft.ipynb             # Microsoft stock analysis
│   ├── stocks.ipynb           # Combined stock analysis
│   ├── models/                 # Pre-trained ML models
│   │   ├── AAPL_model.pkl
│   │   ├── AMZN_model.pkl
│   │   ├── KO_model.pkl
│   │   └── MSFT_model.pkl
│   └── stock_data/            # Historical price data
│       ├── AAPL_price_data.csv
│       ├── AMZN_price_data.csv
│       ├── KO_price_data.csv
│       └── MSFT_price_data.csv
├── trading_view_app/           # Web-based trading dashboard
│   ├── app.py                 # Flask web application
│   ├── index.html             # Dashboard interface
│   └── styles.css             # Dashboard styling
├── tutorial/                   # Learning materials
│   └── random_forest_tutorial.ipynb
└── data/                      # Additional data files
    ├── price_data.csv
    └── trading_data.csv
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Algorithmic-Trading
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import pandas, numpy, matplotlib, sklearn, joblib; print('All dependencies installed successfully')"
   ```

## Usage

### Running the Trading Simulation

1. **Basic trading simulation**
   ```bash
   python daily_trades.py
   ```

2. **Portfolio visualization**
   ```bash
   python graph.py
   ```

3. **Web dashboard**
   ```bash
   cd trading_view_app
   python app.py
   ```
   Then open `http://localhost:5000` in your browser.

### Jupyter Notebooks

- Open individual stock analysis notebooks in `stock_notebooks/`
- Use `tutorial/random_forest_tutorial.ipynb` to learn about the ML approach
- Run `get_money.ipynb` for comprehensive trading analysis

## Configuration

### Trading Parameters

The system uses configurable thresholds for trading decisions:

```python
P_UP_THRESHOLD = 0.6      # Probability threshold for buying
P_DOWN_THRESHOLD = 0.6    # Probability threshold for selling
INITIAL_CAPITAL = 1000    # Starting capital in USD
```

### Model Features

The ML models use the following technical indicators:
- RSI: Relative Strength Index
- k_percent: Stochastic K%
- r_percent: Stochastic R%
- Price_Rate_Of_Change: Price momentum
- MACD: Moving Average Convergence Divergence
- On Balance Volume: Volume-based indicator

## Performance Metrics

The system tracks and reports:
- Total portfolio value over time
- Individual stock performance
- Buy/sell transaction history
- Return on investment (ROI)
- Risk-adjusted returns

## Technical Details

### Machine Learning Models
- **Algorithm**: Random Forest Classifier
- **Features**: 6 technical indicators
- **Output**: Binary classification (price up/down)
- **Training**: Historical data with rolling window approach

### Data Sources
- Historical price data from multiple sources
- Technical indicators calculated using TA-Lib
- Real-time data integration capabilities

### Risk Management
- Position sizing based on confidence levels
- Diversification across multiple stocks
- Stop-loss and take-profit mechanisms
- Capital allocation limits

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It is not intended to provide financial advice or recommendations. Trading involves risk, and past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## Support

For questions or support:
- Open an issue on GitHub
- Review the tutorial notebooks
- Check the documentation in individual modules

## Future Enhancements

- Real-time data integration
- Additional ML algorithms (LSTM, Transformer models)
- Advanced risk management features
- Multi-asset class support
- Cloud deployment options
- API endpoints for external integrations
