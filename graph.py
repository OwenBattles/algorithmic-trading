import pandas as pd
import joblib
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

def load_stock_data():
    """Load stock data with error handling"""
    data_path = "stock_notebooks/stock_data/"
    data = {}
    
    try:
        data['AAPL'] = pd.read_csv(os.path.join(data_path, "AAPL_price_data.csv"))
        data['AMZN'] = pd.read_csv(os.path.join(data_path, "AMZN_price_data.csv"))
        data['KO'] = pd.read_csv(os.path.join(data_path, "KO_price_data.csv"))
        data['MSFT'] = pd.read_csv(os.path.join(data_path, "MSFT_price_data.csv"))
        return data
    except FileNotFoundError as e:
        print(f"Error loading stock data: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading stock data: {e}")
        return None

def run_trading_simulation(models, data, initial_capital=10000):
    """Run trading simulation with proper date handling"""
    if not models or not data:
        print("Cannot run simulation: models or data not loaded")
        return None, None
    
    # Find common date range across all datasets
    common_dates = set()
    for symbol, df in data.items():
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            common_dates.update(df['date'].dt.date)
    
    if not common_dates:
        print("No common dates found across datasets")
        return None, None
    
    # Sort dates and filter to reasonable range
    sorted_dates = sorted(list(common_dates))
    if len(sorted_dates) > 100:  # Limit to last 100 days if too many
        sorted_dates = sorted_dates[-100:]
    
    capital = initial_capital
    portfolio_value = []
    dates_used = []
    
    print(f"Running simulation for {len(sorted_dates)} trading days...")
    
    for date in sorted_dates:
        try:
            # Get predictions for each stock
            total_investment = 0
            
            for symbol in ['AAPL', 'AMZN']:  # Focus on main stocks for this example
                if symbol in models and symbol in data:
                    # Find data for this date
                    date_data = data[symbol][data[symbol]['date'].dt.date == date]
                    if not date_data.empty:
                        # Use first row if multiple entries for same date
                        row = date_data.iloc[0]
                        
                        # Check if required features exist
                        required_features = ['RSI', 'k_percent', 'r_percent', 'Price_Rate_Of_Change', 'MACD', 'On Balance Volume']
                        if all(feature in row.index for feature in required_features):
                            features = [row[feature] for feature in required_features]
                            
                            # Get prediction
                            prediction = models[symbol].predict([features])[0]
                            
                            # Simple strategy: Buy if prediction is 1, sell if 0
                            if 'open' in row.index:
                                if prediction == 1 and capital > 0:
                                    # Buy with 10% of available capital
                                    investment = min(capital * 0.1, capital)
                                    capital -= investment
                                    total_investment += investment
                                elif prediction == 0 and total_investment > 0:
                                    # Sell and add back to capital
                                    capital += total_investment * 0.1
                                    total_investment *= 0.9
            
            # Track portfolio value
            portfolio_value.append(capital + total_investment)
            dates_used.append(date)
            
        except Exception as e:
            print(f"Error processing date {date}: {e}")
            continue
    
    return dates_used, portfolio_value

def plot_portfolio_performance(dates, portfolio_values, initial_capital):
    """Plot portfolio performance over time"""
    if not dates or not portfolio_values:
        print("No data to plot")
        return
    
    plt.figure(figsize=(12, 8))
    plt.plot(dates, portfolio_values, linewidth=2, color='blue')
    plt.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.7, label=f'Initial Capital: ${initial_capital:,.2f}')
    
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    try:
        plt.savefig("portfolio_performance.png", dpi=300, bbox_inches='tight')
        print("Portfolio performance chart saved as 'portfolio_performance.png'")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.show()

def main():
    """Main function to run the trading simulation and visualization"""
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
    
    initial_capital = 10000
    print(f"Starting trading simulation with ${initial_capital:,.2f} initial capital...")
    
    dates, portfolio_values = run_trading_simulation(models, data, initial_capital)
    
    if dates and portfolio_values:
        print(f"Simulation completed. Final portfolio value: ${portfolio_values[-1]:,.2f}")
        plot_portfolio_performance(dates, portfolio_values, initial_capital)
    else:
        print("Simulation failed to produce results.")

if __name__ == "__main__":
    main()
