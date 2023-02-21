"""
 Calculates the average of a selected range of prices, usually closing prices, by the number of
 periods in that range.
"""

import numpy as np
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


def calculate_tr(stock_data, tr_attribute_name):
    """ https: // www.investopedia.com/terms/a/atr.asp """

    # Add a new column called 'tr' filled with zeros
    stock_data[tr_attribute_name] = 0

    for stock in list_stocks(stock_data):
        high = stock_data.loc[stock_data['Stock'] == stock, 'Norm_Adj_Low']
        low = stock_data.loc[stock_data['Stock'] == stock, 'Norm_Adj_Low']
        close = stock_data.loc[stock_data['Stock'] == stock, 'Norm_Adj_Close']
        tr = stock_data.loc[stock_data['Stock'] == stock, tr_attribute_name]

        n = high.shape[0]
        tr[0] = high[0] - low[0]

        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))

        stock_data.loc[stock_data['Stock'] == stock, tr_attribute_name] = tr

    return stock_data


def calculate_atr(stock_data, tr_name, atr_name, period):
    """ https: // www.investopedia.com/terms/a/atr.asp """
    stock_data[atr_name] = 0

    # create datafram to hold new stock_data
    combined_df = pd.DataFrame()

    for stock in list_stocks(stock_data):
        # Filter the dataframe to include only rows where the 'stock' column is the selected stock
        subdata = stock_data[stock_data['Stock'] == stock]

        # Create variable to remember the previous ATR
        prev_atr = 0

        # Iterate over the rows of the dataframe
        for i, row in subdata.iterrows():
            # If this is row 1 to N-1, set the Average True Range to 0
            if i == subdata.index[0]:
                atr = row[tr_name]

            elif i in subdata.index[1:period - 1]:
                atr += row[tr_name]

            # if this is row N, calculate ATR using the first N TR values.
            elif i == subdata.index[period]:
                atr += row[tr_name]
                atr /= period
                subdata.at[i, atr_name] = atr

            # If there is a previous ATR calculated
            else:
                atr = (prev_atr + row[tr_name]) / period
                subdata.at[i, atr_name] = atr

            prev_atr = atr

        combined_df = pd.concat([combined_df, subdata])
    return combined_df


def calculate_adx(stock_data, tr_attribute_name, adx_name, period):
    """ https://www.investopedia.com/terms/w/wilders-dmi-adx.asp """

    # Add a new column called 'adx' filled with zeros
    stock_data[adx_name] = 0

    for stock in list_stocks(stock_data):
        high = stock_data.loc[stock_data['Stock'] == stock, 'Norm_Adj_Low']
        low = stock_data.loc[stock_data['Stock'] == stock, 'Norm_Adj_Low']
        tr = stock_data.loc[stock_data['Stock'] == stock, tr_attribute_name]

        n = high.shape[0]

        dm_plus = np.zeros(n)
        dm_minus = np.zeros(n)
        for i in range(1, n):
            dm_plus[i] = max(0, high[i] - high[i - 1]) \
                if high[i] - high[i - 1] > low[i - 1] - low[i] else 0
            dm_minus[i] = max(0, low[i - 1] - low[i]) \
                if high[i] - high[i - 1] < low[i - 1] - low[i] else 0

        dm_plus_sum = np.zeros(n)
        dm_minus_sum = np.zeros(n)
        for i in range(1, n):
            dm_plus_sum[i] = dm_plus_sum[i - 1] + dm_plus[i]
            dm_minus_sum[i] = dm_minus_sum[i - 1] + dm_minus[i]

        tr_sum = np.zeros(n)
        for i in range(1, n):
            tr_sum[i] = tr_sum[i - 1] + tr[i]

        dx = np.zeros(n)
        for i in range(1, n):
            dx[i] = 100 * (dm_plus_sum[i] / tr_sum[i] - dm_minus_sum[i] / tr_sum[i]) / \
                (dm_plus_sum[i] / tr_sum[i] + dm_minus_sum[i] / tr_sum[i])

        adx = np.zeros(n)
        for i in range(1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        stock_data.loc[stock_data['Stock'] == stock, adx_name] = adx

    return stock_data


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
    
    stock_data = calculate_tr(stock_data, 'tr')

    period = 14
    attribute_name = str(period) + '_day_'
    stock_data = calculate_atr(stock_data, 'tr',  attribute_name + 'atr', period)
    stock_data = calculate_adx(stock_data, 'tr',  attribute_name + 'adx', period)

    return stock_data
