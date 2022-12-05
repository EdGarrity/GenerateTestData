"""
This program downloads stock data, normalizes the data, and saves it to a MS SQL database

1. Write a function in Python called load_index_data that takes a list of index symbols and a date range and gets historical OHLC data from Yahoo! Finance for each index for given date range.
    - Use yfinance
	- adjust all OHLC
		- Set 'Adj_Open' = 'Open' / 'Close' * 'Adj Close'
		- Set 'Adj_High' = 'High' / 'Close' * 'Adj Close'
		- Set 'Adj_Low'  = 'Low'  / 'Close' * 'Adj Close'
		- Set 'Adj_Volume'  = 'Volume'  / 'Adj Close' * 'Close'
    - Normalize 'Adj_Close', 'Adj_High', 'Adj_Low', 'Adj_Open', 'Adj_Volume' between 0.0 and 1.0 and Prefix normalized data with "Norm_"
    - Return the data
2. Write a function in Python called get_configuration_parameters that reads the user_id, password, server, and database from the "sql database" stanza in a configuration file.
	- If file does not exists, create it using the following values as the default values
		- user_id = 'egarrity'
		- password = 'test'
		- server = 'local'
		- database = 'SOS'
3. Write a function in Python called save_to_sql that saves the data to a MS SQL table
    - Create a connection_url using the user_id, password, server, and database from function get_configuration_parameters
    - Create a connection to a MS SQL database using the connection_url
	- Erase all rows from the the MS SQL table StockData if it already exist
    - Save all data to MS SQL table StockData in the database
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sqlalchemy
import os
import configparser


def load_index_data(index_symbols, start_date, end_date):
    """
    This function downloads stock data, normalizes the data, and saves it to a MS SQL database
    """
    # Download data from Yahoo! Finance
    data = yf.download(index_symbols, start=start_date, end=end_date)

    # Adjust all OHLC
    data['Adj_Open'] = data['Open'] / data['Close'] * data['Adj Close']
    data['Adj_High'] = data['High'] / data['Close'] * data['Adj Close']
    data['Adj_Low'] = data['Low'] / data['Close'] * data['Adj Close']
    data['Adj_Volume'] = data['Volume'] / data['Adj Close'] * data['Close']

    # Normalize 'Adj_Close', 'Adj_High', 'Adj_Low', 'Adj_Open', 'Adj_Volume' between 0.0 and 1.0 and Prefix normalized data with "Norm_"
    data['Norm_Adj_Close'] = (data['Adj Close'] - data['Adj Close'].min()) / \
        (data['Adj Close'].max() - data['Adj Close'].min())
    data['Norm_Adj_High'] = (data['Adj_High'] - data['Adj_High'].min()) / \
        (data['Adj_High'].max() - data['Adj_High'].min())
    data['Norm_Adj_Low'] = (data['Adj_Low'] - data['Adj_Low'].min()) / \
        (data['Adj_Low'].max() - data['Adj_Low'].min())
    data['Norm_Adj_Open'] = (data['Adj_Open'] - data['Adj_Open'].min()) / \
        (data['Adj_Open'].max() - data['Adj_Open'].min())
    data['Norm_Adj_Volume'] = (data['Adj_Volume'] - data['Adj_Volume'].min()) / \
        (data['Adj_Volume'].max() - data['Adj_Volume'].min())

    # Return the data
    return data


def get_configuration_parameters():
	"""
	This function reads the user_id, password, server, and database from the "sql database" stanza in a configuration file.
	"""
	# If file does not exists, create it using the following values as the default values
	if not os.path.exists('config.ini'):
		with open('config.ini', 'w') as f:
			f.write('[sql database]\n')
			f.write('user_id = egarrity\n')
			f.write('password = test\n')
			f.write('server = local\n')
			f.write('database = SOS\n')

	# Read the user_id, password, server, and database from the "sql database" stanza in a configuration file
	config = configparser.ConfigParser()
	config.read('config.ini')
	user_id = config['sql database']['user_id']
	password = config['sql database']['password']
	server = config['sql database']['server']
	database = config['sql database']['database']

	# Return the user_id, password, server, and database
	return user_id, password, server, database


def save_to_sql(data):
    """
    This function saves the data to a MS SQL table
    """
    # Create a connection_url using the user_id, password, server, and database from function get_configuration_parameters
    user_id, password, server, database = get_configuration_parameters()
    connection_url = 'mssql+pyodbc://' + user_id + ':' + password + '@' + \
        server + '/' + database + '?driver=SQL+Server+Native+Client+11.0'

    print(connection_url)
    
    # Create a connection to a MS SQL database using the connection_url
    engine = sqlalchemy.create_engine(connection_url)

    # Erase all rows from the the MS SQL table StockData if it already exist
    #  engine.execute('DELETE FROM StockData')

    # Save all data to MS SQL table StockData in the database
    data.to_sql('StockData', engine, if_exists='replace')


# Main
if __name__ == '__main__':
    # Download data from Yahoo! Finance
    data = load_index_data(['AAPL'],
                           '2020-01-01', '2020-12-31')

    # Save all data to MS SQL table StockData in the database
    save_to_sql(data)
