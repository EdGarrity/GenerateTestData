"""
Table stock_data, columns = [Stock, Date, Close]
Table test_data, columns = [Stock, Date, Key, Value]

1. Write a function in Python called populate_stock that takes two pandas_datareader objects:
   stock_data table and test_data
	1. For each record in stock_data:
		1. Create a new record and populate the fields of the new record as follows:
			1. Set the Stock field of the new record to the value of the Stock field in the stock_data table
			2. Set the Date field of the new record to the value of the Date field in the stock_data table
			3. Set the Key field of the new record to 'Close'
			4. Set the Value field of the new record to the value of the Close field in the  table stock_data
		2. Append the new record to the test_data table
	2. return test_data
2. Write a function in Python called test_populate_stock that does the following:
	1. Creates a stock_data table with test data
	2. Creates an emtpy test_data table
	2. Call populate_stock
"""

import configparser
import pandas as pd
import sqlalchemy


def list_columns(dataframe):
    """
    Lists all the columns found in a pandas data record except for the specified columns

    Args:
        pandas data record

    Returns:
        list of found columns

    This function takes a pandas dataframe as input and returns a list of the columns in the
    dataframe that are not in the list of excluded columns. The list of excluded columns is
    hardcoded in the function, so if you want to exclude different columns, you will need to
    modify this list.

    To use this function, you would pass a pandas dataframe as an argument when calling the
    function, like this:

        columns = list_columns(dataframe)
    """

    # excluded_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Symbol', \
    #                     'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Volume']
    excluded_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']
    all_columns = dataframe.columns
    columns = [column for column in all_columns if column not in excluded_columns]
    return columns

def load_stock_data(stock_data, test_data):
    """
    This function the stock data into the KV Collection
    """

    records = []

    for col_name in list_columns(stock_data):
        for index, row in stock_data.iterrows():
            # write open
            new_record = [row['Symbol'], row.name, col_name, row[col_name]]
            records.append(new_record)

    test_data = pd.DataFrame.from_records(
        records, columns=['Symbol', 'Date', 'Key', 'Value'])

    return test_data

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
    connection_url = 'mssql+pyodbc://' + user_id + ':' + password + '@' + \
        server + '/' + database + '?driver=SQL+Server+Native+Client+11.0'

    # create connection
    engine = sqlalchemy.create_engine(connection_url)

    # erase all rows from the the MS SQL table StockData if it already exist
    engine.execute('DELETE FROM TestData')

    # save all data to MS SQL table StockData in the database
    data.to_sql('TestData', engine, if_exists='append')
