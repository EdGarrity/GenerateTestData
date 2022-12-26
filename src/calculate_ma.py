"""
 Calculates the average of a selected range of prices, usually closing prices, by the number of periods in that range.
"""

import pandas as pd

def calculate_simple_moving_average(stock_data, test_data, name, period):
    """ Generates the simple moving averages.  Calculates the average of a range of prices by the number of periods within that range.

     Args:
         stock_data (dataframe): 
         name (string):
         period (int):
    """
    stock_data[name] = stock_data['Norm_Adj_Close'].rolling(period).mean()

    records = []
    
    for index, row in stock_data.iterrows():
        # calculate SMA
        new_record = [row['Stock'], row.name, name, row['Norm_Adj_Open']]
        records.append(new_record)

    return test_data

def generate_technical_indicators(stock_data, test_data):
    test_data = calculate_simple_moving_average(stock_data, test_data, '50_day_ma', 50)
    test_data = calculate_simple_moving_average(stock_data, test_data, '200_day_ma', 200)

    return test_data
