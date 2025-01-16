import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load models
model_path = "stock_notebooks/models/"
aapl_model = joblib.load(model_path + "AAPL_model.pkl")
amzn_model = joblib.load(model_path + "AMZN_model.pkl")
ko_model = joblib.load(model_path + "KO_model.pkl")
msft_model = joblib.load(model_path + "MSFT_model.pkl")

# Buying strategy based on the certainty of tomorrow's market fluctuations
p_up_threshold = 0.6
p_down_threshold = 0.6

capital = 1000

aapl_holdings = 0
amzn_holdings = 0
ko_holdings = 0
msft_holdings = 0

aapl_portfolio_value = []
amzn_portfolio_value = []
ko_portfolio_value = []
msft_portfolio_value = []
portfolio_value = []
stock_shares = []

sample_size = 75

# Load data
data_path = "stock_notebooks/stock_data/"
aapl_data = pd.read_csv(data_path + "AAPL_price_data.csv").tail(sample_size)
amzn_data = pd.read_csv(data_path + "AMZN_price_data.csv").tail(sample_size)
ko_data = pd.read_csv(data_path + "KO_price_data.csv").tail(sample_size)
msft_data = pd.read_csv(data_path + "MSFT_price_data.csv").tail(sample_size)

# Define the feature columns
features = [
    "RSI",
    "k_percent",
    "r_percent",
    "Price_Rate_Of_Change",
    "MACD",
    "On Balance Volume",
]

aapl_close_data = aapl_data["close"]
amzn_close_data = amzn_data["close"]
ko_close_data = ko_data["close"]
msft_close_data = msft_data["close"]


def main():
    global capital, aapl_holdings, amzn_holdings, ko_holdings, msft_holdings

    for day in range(sample_size):
        # define row for each day
        aapl_today = aapl_data.iloc[-sample_size + day]
        amzn_today = amzn_data.iloc[-sample_size + day]
        ko_today = ko_data.iloc[-sample_size + day]
        msft_today = msft_data.iloc[-sample_size + day]

        # Get the prediction probabilities for each day
        aapl_probs = aapl_model.predict_proba([aapl_today[features]])[0]
        amzn_probs = amzn_model.predict_proba([amzn_today[features]])[0]
        ko_probs = ko_model.predict_proba([ko_today[features]])[0]
        msft_probs = msft_model.predict_proba([msft_today[features]])[0]

        # Apply the trading strategy
        aapl_action, aapl_investment = trade_strategy(
            capital / 4, aapl_probs[0], aapl_probs[1], aapl_holdings
        )
        amzn_action, amzn_investment = trade_strategy(
            capital / 4, amzn_probs[0], amzn_probs[1], amzn_holdings
        )
        ko_action, ko_investment = trade_strategy(
            capital / 4, ko_probs[0], ko_probs[1], ko_holdings
        )
        msft_action, msft_investment = trade_strategy(
            capital / 4, msft_probs[0], msft_probs[1], msft_holdings
        )

        # Update capital and holdings based on actions
        capital, aapl_holdings = execute_trade(
            aapl_action,
            aapl_investment,
            aapl_holdings,
            capital,
            aapl_today["close"],
        )
        capital, amzn_holdings = execute_trade(
            amzn_action,
            amzn_investment,
            amzn_holdings,
            capital,
            amzn_today["close"],
        )
        capital, ko_holdings = execute_trade(
            ko_action, ko_investment, ko_holdings, capital, ko_today["close"]
        )
        capital, msft_holdings = execute_trade(
            msft_action,
            msft_investment,
            msft_holdings,
            capital,
            msft_today["close"],
        )

        # Track portfolio value
        aapl_portfolio_value.append(aapl_holdings * aapl_close_data.iloc[day])
        amzn_portfolio_value.append(amzn_holdings * amzn_close_data.iloc[day])
        ko_portfolio_value.append(ko_holdings * ko_close_data.iloc[day])
        msft_portfolio_value.append(msft_holdings * msft_close_data.iloc[day])
        portfolio_value.append(
            aapl_holdings * aapl_close_data.iloc[day]
            + amzn_holdings * amzn_close_data.iloc[day]
            + ko_holdings * ko_close_data.iloc[day]
            + msft_holdings * msft_close_data.iloc[day]
            + capital
            - 1000
        )
        stock_shares.append(
            aapl_close_data.iloc[day]
            + amzn_close_data.iloc[day]
            + ko_close_data.iloc[day]
            + msft_close_data.iloc[day]
        )

        print(ko_holdings)

    # Plot the portfolio value over time
    plot_portfolio_values()


def trade_strategy(p, p_down, p_up, holdings):
    action = "Hold"
    investment = 0

    if p_up > 0.8:
        action = "Buy"
        investment = 0.3 * p
    elif p_up > p_up_threshold:
        action = "Buy"
        investment = 0.1 * p
    elif p_down > 0.8:
        action = "Sell"
        investment = 0.3 * holdings  # Sell 30% of holdings
    elif p_down > p_down_threshold:
        action = "Sell"
        investment = 0.1 * holdings  # Sell 10% of holdings

    return action, investment


def execute_trade(action, investment, holdings, capital, close_price):
    if action == "Buy":
        shares_bought = investment / close_price
        holdings += shares_bought
        capital -= investment
    elif action == "Sell":
        shares_sold = investment / close_price
        holdings -= shares_sold
        capital += investment
    return capital, holdings


def plot_portfolio_values():
    dates = pd.date_range(start="2024-01-01", end="2024-08-29")

    plt.figure(figsize=(14, 7))
    plt.plot(
        dates[: len(aapl_portfolio_value)],
        aapl_portfolio_value,
        label="AAPL Portfolio Value",
    )
    plt.plot(
        dates[: len(amzn_portfolio_value)],
        amzn_portfolio_value,
        label="AMZN Portfolio Value",
    )
    plt.plot(
        dates[: len(ko_portfolio_value)], ko_portfolio_value, label="KO Portfolio Value"
    )
    plt.plot(
        dates[: len(msft_portfolio_value)],
        msft_portfolio_value,
        label="MSFT Portfolio Value",
    )
    plt.plot(
        dates[: len(portfolio_value)],
        portfolio_value,
        label="Total Portfolio Value",
    )

    plt.plot(
        dates[: len(portfolio_value)],
        stock_shares,
        label="Stock shares",
    )

    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
