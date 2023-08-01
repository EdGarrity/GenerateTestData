import pandas as pd

def add_trading_days_in_year(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the `TradingDaysInYear` for each stock in the DataFrame.

    Parameters:
        stock_data (pd.DataFrame): A DataFrame with stock data.

    Returns:
        pd.DataFrame: The DataFrame with the `TradingDaysInYear` field added.
    """
    # Group data by stock symbol and year
    grouped_data = stock_data.groupby([stock_data.index.year, 'Symbol'])

    # Calculate TradingDaysInYear for each stock and year
    stock_data['TradingDaysInYear'] = grouped_data['Adj Close'].transform(lambda x: x.notnull().sum())

    return stock_data