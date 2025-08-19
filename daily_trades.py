import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

def load_models():
    """Load pre-trained models with error handling"""
    model_path = "stock_notebooks/models/"
    models = {}
    
    try:
        models['AAPL'] = joblib.load(os.path.join(model_path, "AAPL_model.pkl"))
        models['AMZN'] = joblib.load(os.path.join(model_path, "AMZN_model.pkl"))
        models['KO'] = joblib.load(os.path.join(model_path, "KO_model.pkl"))
        models['MSFT'] = joblib.load(os.path.join(model_path, "MSFT_model.pkl"))
        return models
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading models: {e}")
        return None

def load_stock_data(sample_size=75):
    """Load stock data with error handling"""
    data_path = "stock_notebooks/stock_data/"
    data = {}
    
    try:
        data['AAPL'] = pd.read_csv(os.path.join(data_path, "AAPL_price_data.csv")).tail(sample_size)
        data['AMZN'] = pd.read_csv(os.path.join(data_path, "AMZN_price_data.csv")).tail(sample_size)
        data['KO'] = pd.read_csv(os.path.join(data_path, "KO_price_data.csv")).tail(sample_size)
        data['MSFT'] = pd.read_csv(os.path.join(data_path, "MSFT_price_data.csv")).tail(sample_size)
        return data
    except FileNotFoundError as e:
        print(f"Error loading stock data: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading stock data: {e}")
        return None

# Trading parameters
P_UP_THRESHOLD = 0.6
P_DOWN_THRESHOLD = 0.6
INITIAL_CAPITAL = 1000

# Feature columns for model prediction
FEATURES = [
    "RSI",
    "k_percent",
    "r_percent",
    "Price_Rate_Of_Change",
    "MACD",
    "On Balance Volume",
]

class TradingSimulator:
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.capital = initial_capital
        self.holdings = {'AAPL': 0, 'AMZN': 0, 'KO': 0, 'MSFT': 0}
        self.portfolio_values = {'AAPL': [], 'AMZN': [], 'KO': [], 'MSFT': []}
        self.total_portfolio_value = []
        self.stock_shares = []
        
    def trade_strategy(self, capital_allocation, p_down, p_up, holdings):
        """Determine trading action based on prediction probabilities"""
        action = "Hold"
        investment = 0

        if p_up > 0.8:
            action = "Buy"
            investment = 0.3 * capital_allocation
        elif p_up > P_UP_THRESHOLD:
            action = "Buy"
            investment = 0.1 * capital_allocation
        elif p_down > 0.8:
            action = "Sell"
            investment = 0.3 * holdings
        elif p_down > P_DOWN_THRESHOLD:
            action = "Sell"
            investment = 0.1 * holdings

        return action, investment

    def execute_trade(self, action, investment, holdings, close_price):
        """Execute a trade and return updated capital and holdings"""
        if action == "Buy":
            shares_bought = investment / close_price
            holdings += shares_bought
            self.capital -= investment
        elif action == "Sell":
            shares_sold = investment / close_price
            holdings -= shares_sold
            self.capital += investment
        return holdings

    def run_simulation(self, models, data, sample_size=75):
        """Run the trading simulation"""
        if not models or not data:
            print("Cannot run simulation: models or data not loaded")
            return
            
        for day in range(sample_size):
            # Get current day data for each stock
            current_data = {}
            for symbol in ['AAPL', 'AMZN', 'KO', 'MSFT']:
                if day < len(data[symbol]):
                    current_data[symbol] = data[symbol].iloc[day]
                else:
                    continue
                    
            # Get predictions for each stock
            for symbol in ['AAPL', 'AMZN', 'KO', 'MSFT']:
                if symbol not in current_data:
                    continue
                    
                try:
                    probs = models[symbol].predict_proba([current_data[symbol][FEATURES]])[0]
                    action, investment = self.trade_strategy(
                        self.capital / 4, probs[0], probs[1], self.holdings[symbol]
                    )
                    
                    # Execute trade
                    self.holdings[symbol] = self.execute_trade(
                        action, investment, self.holdings[symbol], current_data[symbol]["close"]
                    )
                    
                    # Track portfolio values
                    stock_value = self.holdings[symbol] * current_data[symbol]["close"]
                    self.portfolio_values[symbol].append(stock_value)
                    
                except (KeyError, IndexError) as e:
                    print(f"Error processing {symbol} on day {day}: {e}")
                    continue
            
            # Calculate total portfolio value
            total_value = sum(self.portfolio_values[symbol][-1] for symbol in ['AAPL', 'AMZN', 'KO', 'MSFT'] if self.portfolio_values[symbol])
            total_value += self.capital - INITIAL_CAPITAL
            self.total_portfolio_value.append(total_value)
            
            # Track stock shares (sum of close prices)
            stock_sum = sum(current_data[symbol]["close"] for symbol in ['AAPL', 'AMZN', 'KO', 'MSFT'] if symbol in current_data)
            self.stock_shares.append(stock_sum)

    def plot_results(self):
        """Plot portfolio performance over time"""
        if not self.total_portfolio_value:
            print("No data to plot")
            return
            
        # Generate date range based on data length
        start_date = datetime.now() - timedelta(days=len(self.total_portfolio_value))
        dates = pd.date_range(start=start_date, end=datetime.now(), periods=len(self.total_portfolio_value))
        
        plt.figure(figsize=(14, 10))
        
        # Plot individual stock portfolio values
        for symbol in ['AAPL', 'AMZN', 'KO', 'MSFT']:
            if self.portfolio_values[symbol]:
                plt.plot(dates[:len(self.portfolio_values[symbol])], 
                        self.portfolio_values[symbol], 
                        label=f"{symbol} Portfolio Value")
        
        # Plot total portfolio value
        plt.plot(dates[:len(self.total_portfolio_value)], 
                self.total_portfolio_value, 
                label="Total Portfolio Value", linewidth=2)
        
        # Plot stock shares
        if self.stock_shares:
            plt.plot(dates[:len(self.stock_shares)], 
                    self.stock_shares, 
                    label="Stock Shares", linestyle='--')
        
        plt.title("Portfolio Performance Over Time")
        plt.xlabel("Date")
        plt.ylabel("Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        self.print_summary()
    
    def print_summary(self):
        """Print trading simulation summary"""
        print("\n" + "="*50)
        print("TRADING SIMULATION SUMMARY")
        print("="*50)
        print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Total Return: ${self.capital - INITIAL_CAPITAL:,.2f}")
        print(f"Return Percentage: {((self.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100):.2f}%")
        print("\nFinal Holdings:")
        for symbol, shares in self.holdings.items():
            print(f"  {symbol}: {shares:.4f} shares")
        print("="*50)

def main():
    """Main function to run the trading simulation"""
    print("Loading trading models...")
    models = load_models()
    if not models:
        print("Failed to load models. Exiting.")
        return
    
    print("Loading stock data...")
    data = load_stock_data()
    if not data:
        print("Failed to load stock data. Exiting.")
        return
    
    print("Starting trading simulation...")
    simulator = TradingSimulator(INITIAL_CAPITAL)
    simulator.run_simulation(models, data)
    simulator.plot_results()

if __name__ == "__main__":
    main()
