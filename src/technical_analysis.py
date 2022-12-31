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

def calculate_sma(stock_data, name, ticker_field, period):
    """ Generates the simple moving averages.  Calculates the average of a range of prices by the
        number of periods within that range.

     Args:
         stock_data (dataframe):
         name (string):
         period (int):
    """
    for stock in list_stocks(stock_data):
        mask = stock_data['Stock'] == stock

        stock_data.loc[mask, name] = stock_data.loc[mask,
                                                    ticker_field].rolling(period).mean()

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

        # Create variable to remember the previous close
        prev_close = 0

        # Create variable to remember the previous OBV
        prev_obv = 0

        # Iterate over the rows of the dataframe
        for i, row in subdata.iterrows():
            # If this is the first row, set the 'obav' value to the 'volume' value
            if i == subdata.index[0]:
                delta = 0

            # If this is not the first row, calculate the 'obv' value
            elif row['Norm_Adj_Close'] > prev_close:
                delta = row['Norm_Adj_Volume']

            elif row['Norm_Adj_Close'] < prev_close:
                delta = 0 - row['Norm_Adj_Volume']

            else:
                delta = 0

            subdata.at[i, 'obv'] = prev_obv + delta
            prev_close = row['Norm_Adj_Close']
            prev_obv = subdata.at[i, 'obv']

        combined_df = pd.concat([combined_df, subdata])

    # Return the modified dataframe
    return combined_df

def calculate_ema(stock_data, name, ticker_field, period):
    """ Generates the exponential moving averages.
    """

    multiplier = 2 / (period + 1)

    # Add a new column called 'obav' filled with zeros
    stock_data[name] = 0

    # create datafram to hold new stock_data
    combined_df = pd.DataFrame()

    for ticker in list_stocks(stock_data):
        # Filter the dataframe to include only rows where the 'stock' column is the selected stock
        subdata = stock_data[stock_data['Stock'] == ticker]

        # Create variable to remember the previous EMA
        prev_ema = 0

        # Iterate over the rows of the dataframe
        for i, row in subdata.iterrows():
            current_ema = row[ticker_field] * multiplier + prev_ema * (1 - multiplier)
            subdata.at[i, name] = current_ema
            prev_ema = current_ema

        combined_df = pd.concat([combined_df, subdata])
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

    stock_data = calculate_obv(stock_data)

    ticker_fields = ['Norm_Adj_Open',
                     'Norm_Adj_High',
                     'Norm_Adj_Low',
                     'Norm_Adj_Close',
                     'Norm_Adj_Volume',
                     'obv']
    periods = [5, 8, 10, 12, 20, 26, 50, 200]

    for ticker in ticker_fields:
        for period in periods:
            attribute_name = str(period) + '_day_' + ticker
            stock_data = calculate_sma(stock_data, attribute_name + '_sma', ticker, period)
            stock_data = calculate_ema(stock_data, attribute_name + '_ema', ticker, period)

    return stock_data
