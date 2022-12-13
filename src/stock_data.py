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

import os
import configparser
import yfinance as yf
import pandas as pd
import numpy as np
import sqlalchemy

def load_index_data(index_symbols, start_date, end_date):
    """
    This function downloads stock data, normalizes the data, and saves it to a MS SQL database
    :param index_symbols: list of index symbols
    :param start_date: start date
    :param end_date: end date
    :return: dataframe
    """
    # download data
    data= pd.DataFrame()
    for stock in index_symbols:
        df = yf.download(stock, start=start_date, end=end_date)
        df['Stock'] = stock
        data = pd.concat([data, df], axis=0)
    
    # adjust all OHLC
    data['Adj_Open'] = data['Open'] / data['Close'] * data['Adj Close']
    data['Adj_High'] = data['High'] / data['Close'] * data['Adj Close']
    data['Adj_Low'] = data['Low'] / data['Close'] * data['Adj Close']
    data['Adj_Volume'] = data['Volume'] / data['Adj Close'] * data['Close']

    # normalize data
    data['Norm_Adj_Close'] = (data['Adj Close'] - data['Adj Close'].min()) / (data['Adj Close'].max() - data['Adj Close'].min())
    data['Norm_Adj_High'] = (data['Adj_High'] - data['Adj_High'].min()) / (data['Adj_High'].max() - data['Adj_High'].min())
    data['Norm_Adj_Low'] = (data['Adj_Low'] - data['Adj_Low'].min()) / (data['Adj_Low'].max() - data['Adj_Low'].min())
    data['Norm_Adj_Open'] = (data['Adj_Open'] - data['Adj_Open'].min()) / (data['Adj_Open'].max() - data['Adj_Open'].min())
    data['Norm_Adj_Volume'] = (data['Adj_Volume'] - data['Adj_Volume'].min()) / (data['Adj_Volume'].max() - data['Adj_Volume'].min())

    return data

def get_configuration_parameters():
    """
    This function reads the user_id, password, server, and database from the "sql database" stanza in a configuration file.
    :return: user_id, password, server, and database
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
    connection_url = 'mssql+pyodbc://' + user_id + ':' + password + '@' + server + '/' + database + '?driver=SQL+Server+Native+Client+11.0'

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
    stock_data = load_index_data(['AAPL', 'FXAIX'], '2020-01-01', '2020-12-31')

    # save data to sql
    save_to_sql(stock_data)
