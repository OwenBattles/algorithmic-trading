import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load your pre-trained models
model_path = "stock_notebooks/models/"

aapl_model = joblib.load(model_path + "AAPL_model.pkl")
amzn_model = joblib.load(model_path + "AMZN_model.pkl")
ko_model = joblib.load(model_path + "KO_model.pkl")
msft_model = joblib.load(model_path + "MSFT_model.pkl")
# Add other models as necessary

# Load daily stock data
data_path = "stock_notebooks/stock_data/"

aapl_data = pd.read_csv(data_path + "AAPL_price_data.csv")
amzn_data = pd.read_csv(data_path + "AMZN_price_data.csv")
ko_data = pd.read_csv(data_path + "KO_price_data.csv")
msft_data = pd.read_csv(data_path + "MSFT_price_data.csv")


# Starting capital
capital = 10000
portfolio_value = []

# Simulate daily trading
for date in pd.date_range(start="2024-01-01", end="2024-08-29"):
    aapl_prediction = aapl_model.predict(aapl_data.loc[aapl_data["date"] == date])
    amzn_prediction = amzn_model.predict(amzn_data.loc[amzn_data["date"] == date])

    # Example strategy: Buy if prediction is 1, sell if prediction is 0
    if aapl_prediction == 1:
        capital -= aapl_data.loc[aapl_data["date"] == date, "open"]  # Buy
    elif aapl_prediction == 0:
        capital += aapl_data.loc[aapl_data["date"] == date, "open"]  # Sell

    # Track portfolio value
    portfolio_value.append(capital)

# Plot portfolio value over time
plt.plot(pd.date_range(start="YYYY-MM-DD", end="YYYY-MM-DD"), portfolio_value)
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.savefig("portfolio_value.png")
