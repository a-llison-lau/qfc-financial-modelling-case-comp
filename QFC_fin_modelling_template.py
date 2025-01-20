import pandas as pd

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

    def initialize_context_with_data(self, file_path: str) -> None:
        historical_data = pd.read_excel(file_path)
        historical_data.columns = ["Day", "HydroCorp", "BrightFuture"]
        self.historical_prices = {
            "HydroCorp": historical_data["HydroCorp"].tolist(),
            "BrightFuture": historical_data["BrightFuture"].tolist(),
        }  


def update_portfolio(curMarket: Market, curPortfolio: Portfolio, context: Context):
    # YOUR TRADING STRATEGY GOES HERE
    pass
    

###SIMULATION###
market = Market()
portfolio = Portfolio()
context = Context()

file_path = "/Users/allisonlau/VSCodeProjects/qfc-financial-modelling-case-comp/Stock Prices - Days 0-365.xlsx"
context.initialize_context_with_data(file_path)

import pdb; pdb.set_trace()

for i in range(365):
    update_portfolio(market, portfolio, context)
    market.updateMarket()

print(portfolio.evaluate(market))


