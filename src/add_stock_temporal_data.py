import yfinance as yf
import pandas as pd

def load_index_data(index_symbols, start_date, end_date):
    """
    This function downloads stock data and normalizes the data.
    :param index_symbols: list of index symbols
    :param start_date: start date
    :param end_date: end date
    :return: dataframe
    """
    # download data
    data= pd.DataFrame()
    for stock in index_symbols:
        ticker_data = yf.download(stock, start=start_date, end=end_date)
        ticker_data['Stock'] = stock
        data = pd.concat([data, ticker_data], axis=0)

    # adjust all OHLC
    data['Adj_Open'] = data['Open'] / data['Close'] * data['Adj Close']
    data['Adj_High'] = data['High'] / data['Close'] * data['Adj Close']
    data['Adj_Low'] = data['Low'] / data['Close'] * data['Adj Close']
    data['Adj_Volume'] = data['Volume'] / data['Adj Close'] * data['Close']

    for stock in index_symbols:
        mask = data['Stock'] == stock

        data.loc[mask, 'Norm_Adj_Close']  \
            = (data.loc[mask, 'Adj Close'] - data.loc[mask, 'Adj_Low'].min()) \
            / (data.loc[mask, 'Adj_High'].max() - data.loc[mask, 'Adj_Low'].min())

        data.loc[mask, 'Norm_Adj_High']  \
            = (data.loc[mask, 'Adj_High'] - data.loc[mask, 'Adj_Low'].min()) \
            / (data.loc[mask, 'Adj_High'].max() - data.loc[mask, 'Adj_Low'].min())

        data.loc[mask, 'Norm_Adj_Low']  \
            = (data.loc[mask, 'Adj_Low'] - data.loc[mask, 'Adj_Low'].min()) \
            / (data.loc[mask, 'Adj_High'].max() - data.loc[mask, 'Adj_Low'].min())

        data.loc[mask, 'Norm_Adj_Open']  \
            = (data.loc[mask, 'Adj_Open'] - data.loc[mask, 'Adj_Low'].min()) \
            / (data.loc[mask, 'Adj_High'].max() - data.loc[mask, 'Adj_Low'].min())

        data.loc[mask, 'Norm_Adj_Volume']  \
            = (data.loc[mask, 'Adj_Volume'] - data.loc[mask, 'Adj_Volume'].min()) \
            / (data.loc[mask, 'Adj_Volume'].max() - data.loc[mask, 'Adj_Volume'].min())

    return data

def add_stock_temporal_data() -> pd.DataFrame:
    """
    This program downloads stock data and adds the `TradingDaysInYear` field.

    For each record in the downloaded stock data, add the following field:
    - TradingDaysInYear: Calculate the number of trading days in the year for each stock.

    Parameters:
        None

    Returns:
        pd.DataFrame: A DataFrame with the stock data including the added temporal-related information.
    """
    # Download data
    stock_data = load_index_data(['AAPL', 'FXAIX'], '2019-01-01', '2021-12-31')

    # Add TradingDaysInYear field
    stock_data = add_trading_days_in_year(stock_data)

    # Return stock data with additional temporal-related information
    return stock_data


def add_trading_days_in_year(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the `TradingDaysInYear` for each stock in the DataFrame.

    Parameters:
        stock_data (pd.DataFrame): A DataFrame with stock data.

    Returns:
        pd.DataFrame: The DataFrame with the `TradingDaysInYear` field added.
    """
    # Group data by stock symbol
    grouped_data = stock_data.groupby('Symbol')

    # Calculate TradingDaysInYear for each stock
    stock_data['TradingDaysInYear'] = grouped_data['Date'].transform(lambda x: len(x))

    return stock_data
