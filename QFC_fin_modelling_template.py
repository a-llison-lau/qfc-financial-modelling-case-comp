import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Market:
    transaction_fee = 0.005
    def __init__(self) -> None:
        self.stocks = {"HydroCorp": 123, "BrightFuture": 456}

    def updateMarket(self):
        #Will be implemented during grading. 
        #This function will update the stock values to their "real" values each day.
        pass
 
class Portfolio:
    def __init__(self) -> None:
        self.shares = {"HydroCorp": 0, "BrightFuture": 0}
        self.cash = 100000

    def evaluate(self, curMarket: Market) -> float:
        """Total portfolio value."""
        valueA = self.shares["HydroCorp"] * curMarket.stocks["HydroCorp"]
        valueB = self.shares["BrightFuture"] * curMarket.stocks["BrightFuture"]

        return valueA + valueB + self.cash

    def sell(self, stock_TSLA: str, sharesToSell: float, curMarket: Market) -> None:
        if sharesToSell <= 0:
            raise ValueError("Number of shares must be positive")

        if sharesToSell > self.shares[stock_TSLA]:
            raise ValueError("Attempted to sell more stock than is available")

        self.shares[stock_TSLA] -= sharesToSell
        self.cash += (1 - Market.transaction_fee) * sharesToSell * curMarket.stocks[stock_TSLA]

    def buy(self, stock_TSLA: str, sharesToBuy: float, curMarket: Market) -> None:
        if sharesToBuy <= 0:
            raise ValueError("Number of shares must be positive")
        
        cost = (1 + Market.transaction_fee) * sharesToBuy * curMarket.stocks[stock_TSLA]
        if cost > self.cash:
            raise ValueError("Attempted to spend more cash than available")

        self.shares[stock_TSLA] += sharesToBuy
        self.cash -= cost

class Context:
    def __init__(self) -> None:
        self.historical_prices = {}
        self.models = {"HydroCorp": None, "BrightFuture": None}
        self.window_size = 7
        self.current_day = 0

    def initialize_context_with_data(self, file_path: str) -> None:
        historical_data = pd.read_excel(file_path)
        historical_data.columns = historical_data.iloc[0]
        historical_data = historical_data[1:]
        historical_data = historical_data.rename(columns={
            "Day": "Day",
            "HydroCorp": "HydroCorp",
            "BrightFuture Rewables": "BrightFuture"
        })
        historical_data = historical_data[["Day", "HydroCorp", "BrightFuture"]]

        # Plotting the data
        plt.figure(figsize=(10, 6))
        plt.plot(historical_data["Day"], historical_data["HydroCorp"], label="HydroCorp", color="blue")
        plt.plot(historical_data["Day"], historical_data["BrightFuture"], label="BrightFuture", color="green")
        plt.xlabel("Day")
        plt.ylabel("Stock Price")
        plt.title("Historical Stock Prices: HydroCorp vs BrightFuture")
        plt.legend()
        plt.grid(True)
        plt.savefig("historical_data.png")

        self.historical_prices = {
            "HydroCorp": historical_data[["Day", "HydroCorp"]],
            "BrightFuture": historical_data[["Day", "BrightFuture"]],
        }
        self.historical_prices["HydroCorp"] = self.historical_prices["HydroCorp"].rename(columns={"HydroCorp": "Price"})
        self.historical_prices["BrightFuture"] = self.historical_prices["BrightFuture"].rename(columns={"BrightFuture": "Price"})


        # Initially, current_day is day 365
        self.current_day = len(self.historical_prices["HydroCorp"]) - 1

    def update(self, curMarket: Market) -> None:
        """ Appends current day stock prices from market to context
        """
        new_row_hydrocorp = pd.DataFrame([{"Day": self.current_day, "Price": curMarket.stocks["HydroCorp"]}])
        self.historical_prices["HydroCorp"] = pd.concat([self.historical_prices["HydroCorp"], new_row_hydrocorp], ignore_index=True)

        new_row_brightfuture = pd.DataFrame([{"Day": self.current_day, "Price": curMarket.stocks["BrightFuture"]}])
        self.historical_prices["BrightFuture"] = pd.concat([self.historical_prices["BrightFuture"], new_row_brightfuture], ignore_index=True)

    def create_features(self) -> None:
        """Creates sliding window features for each stock.
        """
        for stock in self.historical_prices.keys():
            data = self.historical_prices[stock].copy()

            # Apply differencing to remove trend
            data["Price_Diff"] = data["Price"].diff()  # Difference between consecutive prices
            data = data.dropna()  # Drop NaN values created by differencing

            # Create lagged features on differenced data
            for i in range(1, self.window_size + 1):
                data[f"Price_Lag_{i}"] = data["Price_Diff"].shift(i)

            # Drop rows with NaN values after creating lagged features
            data = data.dropna()

            # Extract features (X) and target (y)
            X = data[[f"Price_Lag_{i}" for i in range(1, self.window_size + 1)]]
            y = data["Price_Diff"]  # We are predicting the difference in price (first differenced)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

            # Train Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions on the test set (predict the difference in price)
            y_pred_diff = model.predict(X_test)

            # To predict the actual price, add the predicted difference to the actual price
            # Assuming the last known price is in the last row of the test set
            last_known_price = self.historical_prices[stock]["Price"].iloc[-1]  # Latest actual price
            predicted_prices = last_known_price + y_pred_diff  # Add predicted price change

            # Correct Plot: Plot predicted actual prices vs actual prices
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test.index, y_test + last_known_price, label="Actual Price", color="blue", marker="o")  # Actual prices
            plt.scatter(y_test.index, predicted_prices, label="Predicted Price", color="red", marker="x")  # Predicted prices
            plt.xlabel("Test Sample Index")
            plt.ylabel("Stock Price")
            plt.title(f"{stock}: Predicted vs Actual Prices (Linear Regression)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{stock}_predicted_vs_actual.png")
            plt.close()  # Close the plot to avoid overlapping figures

            # Calculate and print Mean Squared Error (MSE)
            mse = mean_squared_error(y_test, y_pred_diff)
            print(f"{stock} Model MSE: {mse:.2f}")

            # Store the trained model
            self.models[stock] = model

    def predict(self) -> dict:
        """Predicts the next day's stock price for each stock.
        Returns: {"HydroCorp": stock_price; "BirghtFuture": stock_price}
        """
        predictions = {}
        for stock in self.historical_prices.keys():
            recent_data = self.historical_prices[stock].iloc[-self.window_size:]["Price"].values

            if len(recent_data) < self.window_size:
                print(f"Not enough data to predict for {stock}")
                predictions[stock] = None
                continue

            recent_data = recent_data.reshape(1, -1)

            model = self.models[stock]
            predicted_price = model.predict(recent_data)[0]
            predictions[stock] = predicted_price

        return predictions

def update_portfolio(curMarket: Market, curPortfolio: Portfolio, context: Context):
    # YOUR TRADING STRATEGY GOES HERE
    if (context.current_day - 365) != 0:
        # Not first day of trading, update market with new stock prices
        context.update(curMarket)

    context.create_features()
    predictions = context.predict()
    
    #TODO: use predictions to do something
    for stock in context.historical_prices.keys():
        prices = context.historical_prices[stock]
        # Ensure there's enough data to calculate the long-term moving average
        if len(prices) >= 10:
            # Calculate short-term and long-term moving averages
            short_term_ma = (prices['Price'][-3:].mean()*3 + predictions[stock])/4
            long_term_ma = (prices['Price'][-10:].mean()*10 + predictions[stock])/11
            
            current_price = curMarket.stocks[stock]
            shares = curPortfolio.shares[stock]

            # If short-term MA crosses above long-term MA, buy; if it crosses below, sell
            if short_term_ma > long_term_ma:
                # Buy if not already bought
                if shares == 0:
                    shares_to_buy = int(curPortfolio.cash / (current_price * (1 + Market.transaction_fee)))
                    if shares_to_buy > 0:
                        curPortfolio.buy(stock, shares_to_buy, curMarket)
            elif short_term_ma < long_term_ma:
                # Sell if shares are held
                if shares > 0:
                    curPortfolio.sell(stock, shares, curMarket)
    
    context.current_day += 1

###SIMULATION###
market = Market()
portfolio = Portfolio()
context = Context()

file_path = "/Users/allisonlau/VSCodeProjects/qfc-financial-modelling-case-comp/Stock Prices - Days 0-365.xlsx"
context.initialize_context_with_data(file_path)

# import pdb; pdb.set_trace()

for i in range(365):
    update_portfolio(market, portfolio, context)
    market.updateMarket()
    # import pdb; pdb.set_trace()

print(portfolio.evaluate(market))


