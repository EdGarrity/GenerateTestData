"""
Table stock_data, columns = [Stock, Date, Close]
Table test_data, columns = [Stock, Date, Key, Value]

1. Write a function in Python called populate_stock that takes two pandas_datareader objects: stock_data table and test_data
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


def ensure_data_frame(fn):
    """
    This function the stock data into the KV Collection
    """
    def wrapper(df1, df2):
        return fn(pd.DataFrame(df1),pd.DataFrame(df2))
    return wrapper


@ensure_data_frame
def load_stock_data(stock_data, test_data):
    """
    This function the stock data into the KV Collection
    """

    records = []
    
    for index, row in stock_data.iterrows():
        # write open
        new_record = [row['Stock'], row.name, 'Open', row['Norm_Adj_Open']]
        records.append(new_record)
    
        # write high
        new_record = [row['Stock'], row.name, 'High', row['Norm_Adj_High']]
        records.append(new_record)
    
        # write low
        new_record = [row['Stock'], row.name, 'Low', row['Norm_Adj_Low']]
        records.append(new_record)
    
        # write close
        new_record = [row['Stock'], row.name, 'Close', row['Norm_Adj_Close']]
        records.append(new_record)

        # write volume
        new_record = [row['Stock'], row.name, 'Volume', row['Norm_Adj_Volume']]
        records.append(new_record)

    test_data = pd.DataFrame.from_records(
        records, columns=['Stock', 'Date', 'Key', 'Value'])
    
    return test_data
 
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
    connection_url = 'mssql+pyodbc://' + user_id + ':' + password + '@' + \
        server + '/' + database + '?driver=SQL+Server+Native+Client+11.0'

    # create connection
    engine = sqlalchemy.create_engine(connection_url)

    # erase all rows from the the MS SQL table StockData if it already exist
    engine.execute('DELETE FROM TestData')

    # save all data to MS SQL table StockData in the database
    data.to_sql('TestData', engine, if_exists='append')
