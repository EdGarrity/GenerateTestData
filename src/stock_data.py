"""
This program downloads stock data, normalizes the data, and saves it to a MS SQL database

1. Write a function in Python called load_index_data that takes a list of index symbols and a date
range and gets historical OHLC data from Yahoo! Finance for each index for given date range.
    - Use yfinance
    - Save stock name and OHLC data in a dataframe
	- adjust all OHLC
		- Set 'Adj_Open' = 'Open' / 'Close' * 'Adj Close'
		- Set 'Adj_High' = 'High' / 'Close' * 'Adj Close'
		- Set 'Adj_Low'  = 'Low'  / 'Close' * 'Adj Close'
		- Set 'Adj_Volume'  = 'Volume'  / 'Adj Close' * 'Close'
    - Normalize 'Adj_Close', 'Adj_High', 'Adj_Low', 'Adj_Open', 'Adj_Volume' between 0.0 and 1.0
    and Prefix normalized data with "Norm_"
    - Return the data
2. Write a function in Python called get_configuration_parameters that reads the user_id, password,
server, and database from the "sql database" stanza in a configuration file.
	- If file does not exists, create it using the following values as the default values
		- user_id = 'egarrity'
		- password = 'test'
		- server = 'local'
		- database = 'SOS'
3. Write a function in Python called save_to_sql that saves the data to a MS SQL table
    - Create a connection_url using the user_id, password, server, and database from function
    get_configuration_parameters
    - Create a connection to a MS SQL database using the connection_url
	- Erase all rows from the the MS SQL table StockData if it already exist
    - Save all data to MS SQL table StockData in the database
"""

import configparser
import yfinance as yf
import pandas as pd
import sqlalchemy

# def load_index_data(index_symbols, start_date, end_date):
#     """
#     This function downloads stock data, normalizes the data, and saves it to a MS SQL database
#     :param index_symbols: list of index symbols
#     :param start_date: start date
#     :param end_date: end date
#     :return: dataframe
#     """
#     # download data
#     data= pd.DataFrame()
#     for stock in index_symbols:
#         ticker_data = yf.download(stock, start=start_date, end=end_date)
#         ticker_data['Stock'] = stock
#         data = pd.concat([data, ticker_data], axis=0)

#     # adjust all OHLC
#     data['Adj_Open'] = data['Open'] / data['Close'] * data['Adj Close']
#     data['Adj_High'] = data['High'] / data['Close'] * data['Adj Close']
#     data['Adj_Low'] = data['Low'] / data['Close'] * data['Adj Close']
#     data['Adj_Volume'] = data['Volume'] / data['Adj Close'] * data['Close']

#     for stock in index_symbols:
#         mask = data['Stock'] == stock

#         data.loc[mask, 'Norm_Adj_Close']  \
#             = (data.loc[mask, 'Adj Close'] - data.loc[mask, 'Adj_Low'].min()) \
#             / (data.loc[mask, 'Adj_High'].max() - data.loc[mask, 'Adj_Low'].min())

#         data.loc[mask, 'Norm_Adj_High']  \
#             = (data.loc[mask, 'Adj_High'] - data.loc[mask, 'Adj_Low'].min()) \
#             / (data.loc[mask, 'Adj_High'].max() - data.loc[mask, 'Adj_Low'].min())

#         data.loc[mask, 'Norm_Adj_Low']  \
#             = (data.loc[mask, 'Adj_Low'] - data.loc[mask, 'Adj_Low'].min()) \
#             / (data.loc[mask, 'Adj_High'].max() - data.loc[mask, 'Adj_Low'].min())

#         data.loc[mask, 'Norm_Adj_Open']  \
#             = (data.loc[mask, 'Adj_Open'] - data.loc[mask, 'Adj_Low'].min()) \
#             / (data.loc[mask, 'Adj_High'].max() - data.loc[mask, 'Adj_Low'].min())

#         data.loc[mask, 'Norm_Adj_Volume']  \
#             = (data.loc[mask, 'Adj_Volume'] - data.loc[mask, 'Adj_Volume'].min()) \
#             / (data.loc[mask, 'Adj_Volume'].max() - data.loc[mask, 'Adj_Volume'].min())

#     return data

def load_index_data(index_Stocks, start_date, end_date):
    """
    This function downloads stock data and normalizes the data.
    :param index_Stocks: list of index Stocks
    :param start_date: start date
    :param end_date: end date
    :return: dataframe
    """
    # download data
    data = pd.DataFrame()
    for stock in index_Stocks:
        ticker_data = yf.download(stock, start=start_date, end=end_date)
        ticker_data['Symbol'] = stock
        data = pd.concat([data, ticker_data], axis=0)

    # Set the DataFrame index to be the row names (date)
    data.index.name = 'Date'

    # adjust all OHLC using the row names (date)
    data['Adj_Open'] = data['Open'] / data['Close'] * data['Adj Close']
    data['Adj_High'] = data['High'] / data['Close'] * data['Adj Close']
    data['Adj_Low'] = data['Low'] / data['Close'] * data['Adj Close']
    data['Adj_Volume'] = data['Volume'] / data['Adj Close'] * data['Close']

    for stock in index_Stocks:
        mask = data['Symbol'] == stock

        # The normalization calculations now use the row names (date) instead of 'Date' column
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

def get_configuration_parameters():
    """
    This function reads the user_id, password, server, and database from the "sql database" stanza
    in a configuration file.

    return: user_id, password, server, and database
    """
    # read configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # get user_id, password, server, and database
    user_id = config['sql database']['user_id']
    password = config['sql database']['password']
    server = config['sql database']['server']
    database = config['sql database']['database']

    return user_id, password, server, database

def save_to_sql(data):
    """
    This function saves the data to a MS SQL table
    :param data: dataframe
    :return:
    """
    # get user_id, password, server, and database
    user_id, password, server, database = get_configuration_parameters()

    # create connection_url
    connection_url = 'mssql+pyodbc://' \
                   + user_id \
                   + ':' \
                   + password \
                   + '@' \
                   + server \
                   + '/' \
                   + database \
                   + '?driver=SQL+Server+Native+Client+11.0'

    # create connection
    engine = sqlalchemy.create_engine(connection_url)

    # erase all rows from the the MS SQL table StockData if it already exist
    engine.execute('DELETE FROM StockData')

    # save all data to MS SQL table StockData in the database
    data.to_sql('StockData', engine, if_exists='append')

def get():
    """
    This program downloads stock data, normalizes the data, and saves it to a MS SQL database
    """
    # download data
    stock_data = load_index_data(['AAPL', 'FXAIX', 'FNCMX'], '2019-01-01', '2021-12-31')

    # save data to sql
    save_to_sql(stock_data)

    return stock_data
