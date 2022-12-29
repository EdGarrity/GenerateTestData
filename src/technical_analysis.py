"""
 Calculates the average of a selected range of prices, usually closing prices, by the number of
 periods in that range.
"""

import pandas as pd

def is_stock_data_empty(data):
    """
    This function takes a pandas dataframe as an input and returns True if the dataframe is None or
    empty, and False otherwise. It does this by using the is operator to check if the dataframe is
    None, and the empty attribute of the dataframe to check if it is empty.

    """

    # Check if the dataframe is None
    if data is None:
        return True

    # Check if the dataframe is empty
    if data.empty:
        return True

    # If the dataframe is not None and not empty, return False
    return False

def list_stocks(data):
    """
    Lists all the unique values of the 'stock' column in a pandas data record

    This function takes a pandas data record as an input and returns a list of the unique values in
    the 'stock' column. It does this by using the unique() method of the pandas Series object, which
    returns a list of the unique values in the series.

    You can then call this function on a pandas data record like this:

        data = pd.read_csv('stock_data.csv')
        list_stocks(data)

    Args:
        data
    """

    # Extract the 'stock' column from the data
    stock_column = data['Stock']

    # Get a list of the unique values in the 'stock' column
    unique_stocks = stock_column.unique()

    # Print the list of unique stocks
    print(unique_stocks)

    return unique_stocks

def sort_data(dataframe):
    """
    Sort data in a pandas dataframe by the index (date)

    Args:
        data to sort

    Returns:
        nothing
    """

    dataframe.sort_index(inplace=True)

def calculate_simple_moving_average(stock_data, name, period):
    """ Generates the simple moving averages.  Calculates the average of a range of prices by the
        number of periods within that range.

     Args:
         stock_data (dataframe):
         name (string):
         period (int):
    """
    for stock in list_stocks(stock_data):
        mask = stock_data['Stock'] == stock

        stock_data.loc[mask, name] = stock_data.loc[mask, 'Norm_Adj_Close'].rolling(period).mean()

    return stock_data

def calculate_obv(stock_data):
    """
    Iterates through a sorted pandas dataframe and sets the 'obv' field as follows:
        OBV = prevOBV + volume,   if close > prev_close
                        0,        if close = prev_close
                        - volume, if close < prev_close
        where:
            OBV = Current on-balance volume level
            prev_OBV = Previous on-balance volume level
            volume = Latest trading volume amount
    """

    # Add a new column called 'obav' filled with zeros
    stock_data['obv'] = 0

    # create datafram to hold new stock_data
    combined_df = pd.DataFrame()

    for stock in list_stocks(stock_data):
        # Filter the dataframe to include only rows where the 'stock' column is the selected stock
        subdata = stock_data[stock_data['Stock'] == stock]

        # Create variable to remember the previous volume
        prev_volume = 0

        #Iterate over the rows of the dataframe
        for i, row in subdata.iterrows():
            # If this is the first row, set the 'obav' value to the 'volume' value
            if i == subdata.index[0]:
                subdata.at[i, 'obv'] = row['Norm_Adj_Volume']
                prev_volume = 0

            # If this is not the first row, set the 'obv' value to the current 'volume' minus the
            # previous 'volume'
            else:
                subdata.at[i, 'obv'] = row['Norm_Adj_Volume'] - prev_volume

        combined_df = pd.concat([combined_df, subdata])

    # Return the modified dataframe
    return combined_df

def generate(stock_data):
    """
    Generate the technical analysis data needed to evaluate the stock information and identify
    trading opportunities in price trends and patterns

    Args:
        stock data

    Returns:
        stock data with technical analysis
    """

    sort_data(stock_data)

    stock_data = calculate_simple_moving_average(stock_data, '50_day_ma', 50)
    stock_data = calculate_simple_moving_average(stock_data, '200_day_ma', 200)
    stock_data = calculate_obv(stock_data)

    return stock_data
