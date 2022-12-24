"""
 Calculates the average of a selected range of prices, usually closing prices, by the number of periods in that range.
"""

import pandas as pd

def calculate_simple_moving_average(stock_data, name, period):
    """ Generates the simple moving averages.  Calculates the average of a range of prices by the number of periods within that range.

     Args:
         stock_data (dataframe): 
         name (string):
         period (int):
    """
    stock_data[name] = stock_data['Norm_Adj_Close'].rolling(period).mean()

    return stock_data
