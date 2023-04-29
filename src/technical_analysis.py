"""
 Calculates the average of a selected range of prices, usually closing prices, by the number of
 periods in that range.
"""

import importlib
import numpy as np
import pandas as pd
import tti

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


def call_lib_function(module_name, function_name, input_data, period=None):
    # Import the module dynamically based on its name
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise ValueError(f"Module {module_name} not found")

    # Get the function object by its name from the imported module
    try:
        function = getattr(module, function_name)
    except AttributeError:
        raise ValueError(
            f"Function {function_name} not found in module {module_name}")

    # Call the function with the specified arguments and keyword arguments
    if (period is None):
        return function(input_data)
    else:
        return function(input_data, period)

def calculate_aggregate(stock_data, period):
    """
    Calculates the following:
        MA
        Min
        Max
        Bias
        DeltaMA
        DeltaBias
        TV
        HighestHigh
        HighestLow
        LowestHigh
        LowestLow

    Args:
        stock_data (pd.DataFrame): A pandas DataFrame with columns "date" and "close".
        period: The date range to aggregate

    Returns:
        pd.DataFrame: data with additional collumns for each aggregate
    """

    attribute_name = str(period) + '_day_'

    for stock in list_stocks(stock_data):
        mask = stock_data['Stock'] == stock

        stock_data.loc[mask, attribute_name + 'MA'] = stock_data.loc[mask,'Norm_Adj_Close'].rolling(window=period).mean()
        stock_data.loc[mask, attribute_name + 'Min'] = stock_data.loc[mask,'Norm_Adj_Close'].rolling(window=period).min()
        stock_data.loc[mask, attribute_name + 'Max'] = stock_data.loc[mask,'Norm_Adj_Close'].rolling(window=period).max()
        stock_data.loc[mask, attribute_name + 'HighestHigh'] = stock_data.loc[mask,'Norm_Adj_High'].rolling(window=period).max()
        stock_data.loc[mask, attribute_name + 'HighestLow'] = stock_data.loc[mask,'Norm_Adj_High'].rolling(window=period).min()
        stock_data.loc[mask, attribute_name + 'LowestHigh'] = stock_data.loc[mask,'Norm_Adj_Low'].rolling(window=period).max()
        stock_data.loc[mask, attribute_name + 'LowestLow'] = stock_data.loc[mask,'Norm_Adj_Low'].rolling(window=period).min()
        stock_data.loc[mask, attribute_name + 'Bias'] = stock_data.loc[mask,'Norm_Adj_Close'] - stock_data.loc[mask, attribute_name + 'MA']
        stock_data.loc[mask, attribute_name + 'TV'] = stock_data.loc[mask,'Norm_Adj_Volume'].rolling(window=period).mean()
        stock_data.loc[mask, attribute_name + 'DeltaMA'] = stock_data.loc[mask, attribute_name + 'MA'] - stock_data.loc[mask, attribute_name + 'MA'].shift(1)
        stock_data.loc[mask, attribute_name + 'DeltaBios'] = stock_data.loc[mask, attribute_name + 'Bias'] - stock_data.loc[mask, attribute_name + 'Bias'].shift(1)

    return stock_data

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


# def calculate_tr(stock_data, tr_attribute_name):
#     """ https: // www.investopedia.com/terms/a/atr.asp """

#     # Add a new column called 'tr' filled with zeros
#     stock_data[tr_attribute_name] = 0

#     for stock in list_stocks(stock_data):
#         high = stock_data.loc[stock_data['Stock'] == stock, 'Norm_Adj_High']
#         low = stock_data.loc[stock_data['Stock'] == stock, 'Norm_Adj_Low']
#         close = stock_data.loc[stock_data['Stock'] == stock, 'Norm_Adj_Close']
#         tr = stock_data.loc[stock_data['Stock'] == stock, tr_attribute_name]

#         n = high.shape[0]
#         tr[0] = high[0] - low[0]

#         for i in range(1, n):
#             tr[i] = max(high[i] - low[i],
#                         abs(high[i] - close[i - 1]),
#                         abs(low[i] - close[i - 1]))

#         stock_data.loc[stock_data['Stock'] == stock, tr_attribute_name] = tr

#     return stock_data


# def calculate_atr(stock_data, tr_name, atr_name, period):
#     """ https: // www.investopedia.com/terms/a/atr.asp """
#     stock_data[atr_name] = 0

#     # create datafram to hold new stock_data
#     combined_df = pd.DataFrame()

#     for stock in list_stocks(stock_data):
#         # Filter the dataframe to include only rows where the 'stock' column is the selected stock
#         subdata = stock_data[stock_data['Stock'] == stock]

#         # Create variable to remember the previous ATR
#         prev_atr = 0

#         # Iterate over the rows of the dataframe
#         for i, row in subdata.iterrows():
#             # If this is row 1 to N-1, set the Average True Range to 0
#             if i == subdata.index[0]:
#                 atr = row[tr_name]

#             elif i in subdata.index[1:period - 1]:
#                 atr += row[tr_name]

#             # if this is row N, calculate ATR using the first N TR values.
#             elif i == subdata.index[period]:
#                 atr += row[tr_name]
#                 atr /= period
#                 subdata.at[i, atr_name] = atr

#             # If there is a previous ATR calculated
#             else:
#                 atr = (prev_atr + row[tr_name]) / period
#                 subdata.at[i, atr_name] = atr

#             prev_atr = atr

#         combined_df = pd.concat([combined_df, subdata])
#     return combined_df


# def calculate_adx(stock_data, tr_attribute_name, adx_name, period):
#     """ https://www.investopedia.com/terms/w/wilders-dmi-adx.asp """

#     # Add a new column called 'adx' filled with zeros
#     stock_data[adx_name] = 0

#     for stock in list_stocks(stock_data):
#         high = stock_data.loc[stock_data['Stock'] == stock, 'Norm_Adj_High']
#         low = stock_data.loc[stock_data['Stock'] == stock, 'Norm_Adj_Low']
#         tr = stock_data.loc[stock_data['Stock'] == stock, tr_attribute_name]

#         n = stock_data.loc[stock_data['Stock'] == stock].shape[0]

#         dm_plus = np.zeros(n)
#         dm_minus = np.zeros(n)
#         for i in range(1, n):
#             dm_plus[i] = max(0, high[i] - high[i - 1]) \
#                 if high[i] - high[i - 1] > low[i - 1] - low[i] else 0
#             dm_minus[i] = max(0, low[i - 1] - low[i]) \
#                 if high[i] - high[i - 1] < low[i - 1] - low[i] else 0

#         dm_plus_sum = np.zeros(n)
#         dm_minus_sum = np.zeros(n)
#         for i in range(1, n):
#             dm_plus_sum[i] = dm_plus_sum[i - 1] + dm_plus[i]
#             dm_minus_sum[i] = dm_minus_sum[i - 1] + dm_minus[i]

#         tr_sum = np.zeros(n)
#         for i in range(1, n):
#             tr_sum[i] = tr_sum[i - 1] + tr[i]

#         dx = np.zeros(n)
#         for i in range(1, n):
#             n1 = dm_plus_sum[i] / tr_sum[i]
#             n2 = dm_minus_sum[i] / tr_sum[i]
#             # print(i, n1, n2, dm_plus_sum[i],  dm_minus_sum[i], tr_sum[i])
            
#             if (n1 + n2 == 0):
#                 dx[1] = 0
#             else:
#                 dx[i] = 100 * (n1 - n2) / (n1 + n2)

#         adx = np.zeros(n)
#         for i in range(1, n):
#             adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

#         stock_data.loc[stock_data['Stock'] == stock, adx_name] = adx

#     return stock_data


def calculate_bb(stock_data, bb_name, window_size=20, num_std_dev=2):
    """ https://www.investopedia.com/terms/b/bollingerbands.asp """

    for stock in list_stocks(stock_data):
        mask = stock_data['Stock'] == stock

        # Calculate the Typical Price
        tp = (stock_data.loc[mask, 'Norm_Adj_High'] +
              stock_data.loc[mask, 'Norm_Adj_Low'] +
              stock_data.loc[mask, 'Norm_Adj_Close']) / 3
            
        # Calculate the rolling mean and standard deviation
        rolling_mean = tp.rolling(window=window_size).mean()
        rolling_std = tp.rolling(window=window_size).std()

        # Calculate the upper and lower bands
        upper_band = rolling_mean + (num_std_dev * rolling_std)
        lower_band = rolling_mean - (num_std_dev * rolling_std)

        bb_upper = bb_name + '_upper'
        bb_lower = bb_name + '_lower'
        bb_ma = bb_name + '_ma'
        
        stock_data.loc[mask, bb_upper] = upper_band
        stock_data.loc[mask, bb_lower] = lower_band
        stock_data.loc[mask, bb_ma] = rolling_mean

    return stock_data


def calculate_rsi(stock_data, name, window_size=14):
    """ https://en.wikipedia.org/wiki/Relative_strength_index """
   
    for stock in list_stocks(stock_data):
        mask = stock_data['Stock'] == stock

        # Calculate the Relative Strength Index (RSI)
        prices = stock_data.loc[mask, 'Norm_Adj_Close']
        deltas = np.diff(prices)
        seed = deltas[:window_size+1]
        up = seed[seed >= 0].sum()/window_size
        down = -seed[seed < 0].sum()/window_size
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:window_size] = 100. - 100./(1.+rs)

        for i in range(window_size, len(prices)):
            delta = deltas[i-1]  # cause the diff is 1 shorter
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(window_size-1) + upval)/window_size
            down = (down*(window_size-1) + downval)/window_size
            rs = up/down
            rsi[i] = 100. - 100./(1.+rs)

        stock_data.loc[mask, name]=rsi

    return stock_data


def calculate_macd(stock_data, signal_period=9):
    """
    The Moving Average Convergence Divergence (MACD) is a technical indicator 
    that shows the relationship between two moving averages of a security’s 
    price. It can help traders time their entries and exits with market 
    momentum.

    To calculate the MACD in Python, you need to import numpy and pandas 
    libraries and use their methods to compute the exponential moving averages 
    (EMA) of the price data. Then you subtract the 26-period EMA from the 
    12-period EMA to get the MACD line. You can also calculate a 9-period EMA 
    of the MACD line to get the signal line, which can be used to generate buy 
    and sell signals.
    
    Args:
        stock_data (pd.DataFrame): A pandas DataFrame with columns "date" and "close".
        signal_period (int, optional): Number of periods used for signal line (default=9).

    Returns:
        pd.DataFrame: data with additional columns "macd" and "signal".
    """

    for stock in list_stocks(stock_data):
        mask = stock_data['Stock'] == stock

        # Calculate MACD line
        stock_data.loc[mask, 'MACD'] = stock_data.loc[mask, '12_day_Norm_Adj_Close_ema'] - stock_data.loc[mask, '26_day_Norm_Adj_Close_ema']

        # Calculate signal line
        stock_data.loc[mask, 'Signal'] = stock_data.loc[mask, 'MACD'].ewm(span=signal_period, adjust=False).mean()

    return stock_data


def stochastic_oscillator(stock_data, attribute_prefix, n=14, d=3):
    """
    The Stochastic Oscillator is a popular technical indicator used to identify
    overbought or oversold conditions in a financial instrument. It is
    calculated using the following formula:

    %K = 100 * (C - L5) / (H5 - L5)

    Where:
	    • C = the most recent closing price
	    • L5 = the lowest price of the last 5 periods
	    • H5 = the highest price of the last 5 periods

    stochastic_oscillator() takes in a dataframe containing the price data for 
    a financial instrument (Open, High, Low, Close), and a parameter n which 
    is the number of periods to use for the calculation. It then calculates the 
    lowest low and highest high for each period using the rolling() function 
    from pandas, and then applies the Stochastic Oscillator formula to compute 
    the %K value for each period.

    The resulting %K values are then added as a new column to the input dataframe.

    Args:
        stock_data (pd.DataFrame): A pandas DataFrame with columns "date" and "close".
        n (int, optional): Number of periods used for signal line (default=9).  Defaults to 5.

    Returns:
        pd.DataFrame: data with additional column %K
    """

    attribute_k_name = attribute_prefix + '%K'
    attribute_d_name = attribute_prefix + '%D'

    for stock in list_stocks(stock_data):
        mask = stock_data['Stock'] == stock

        # Calculate Stochastic Oscillator for a given dataframe
        lowest_low = stock_data.loc[mask,'Norm_Adj_Low'].rolling(window=n).min()
        highest_high = stock_data.loc[mask,'Norm_Adj_High'].rolling(window=n).max()

        k_percent = 100 * \
            (stock_data.loc[mask, 'Norm_Adj_Close'] -
             lowest_low) / (highest_high - lowest_low)

        d_percent = k_percent.rolling(window=d).mean()
        
        stock_data.loc[mask, attribute_k_name] = k_percent
        stock_data.loc[mask, attribute_d_name] = d_percent

    return stock_data


def calculate_rps(stock_data, benchmark_ticker='FXAIX'):
    """ Compares the performance of a stock to its benchmark index over time

    Args:
        stock_data (_type_): A pandas DataFrame with columns "date" and "close".
        reference_ticker (str, optional): Benchmark ticker symbol. Defaults to 'FXAIX'.

    Returns:
        pd.DataFrame: data with additional RPS column
    """

    attribute_name = benchmark_ticker + '_rps'

    stock_data[attribute_name] = 0
    
    for ticker in list_stocks(stock_data):
        # if (ticker != benchmark_ticker):
        ticker_mask = stock_data['Stock'] == ticker
        benchmark_ticker_mask = stock_data['Stock'] == benchmark_ticker
        
        stock_data.loc[ticker_mask, attribute_name] = stock_data.loc[ticker_mask, 'Norm_Adj_Close'] / \
            stock_data.loc[benchmark_ticker_mask, 'Norm_Adj_Close']
    
    stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    stock_data.fillna(method='ffill', inplace=True)

    return stock_data

def calculate_stoch_rsi (stock_data, period=14, k=3, d=3):
    """
    Calculates the Stochastic RSI (StochRSI)

    It calculates the Stochastic RSI by normalizing the RSI values over the 
    specified period and multiplying by 100. Then, it calculates a simple 
    moving average of the Stochastic RSI over the specified smoothing period.
    
    Args:
        stock_data (_type_): A pandas DataFrame with columns "date" and "close".
        period (int, optional): The period for the RSI. Defaults to 14.
        k (int, optional): The period for the Stochastic RSI. Defaults to 3.
        d (int, optional): The period for the Stochastic RSI smoothing. Defaults to 3.

    Returns:
        pd.DataFrame: data with additional RSI column
    """
    rsi_attribute_name = str(period) + '_day_rsi'
    stoch_rsi_attribute_name = str(period) + '_day_stoch_rsi'

    for ticker in list_stocks(stock_data):
        ticker_mask = stock_data['Stock'] == ticker
        
        stoch_rsi = ((stock_data.loc[ticker_mask, rsi_attribute_name] \
                      - stock_data.loc[ticker_mask, rsi_attribute_name].rolling(k).min()) \
                     / (stock_data.loc[ticker_mask, rsi_attribute_name].rolling(k).max() \
                        - stock_data.loc[ticker_mask, rsi_attribute_name].rolling(k).min()) \
                     ) * 100
        stoch_rsi_d = stoch_rsi.rolling(d).mean()
        stock_data.loc[ticker_mask, stoch_rsi_attribute_name] = stoch_rsi_d
    
    return stock_data
        

def calculate_atr(stock_data, period=14):
    """
    The Average True Range (ATR) is a technical analysis indicator that 
    measures market volatility by calculating the moving average of the true 
    range over a specified period1. The true range is defined as the greatest 
    of the following:

    The difference between the current high and the current low
    The absolute value of the difference between the previous close and the current high
    The absolute value of the difference between the previous close and the current low

    Args:
        stock_data (_type_): A pandas DataFrame with columns "date" and "close".
        period (int, optional): The period for the RSI. Defaults to 14.

    Returns:
        pd.DataFrame: data with additional TR and ATR columns
    """
    tr_attribute_name = str(period) + '_day_tr'
    atr_attribute_name = str(period) + '_day_atr'

    for ticker in list_stocks(stock_data):
        ticker_mask = stock_data['Stock'] == ticker

        high = stock_data.loc[ticker_mask, 'Norm_Adj_High']
        low = stock_data.loc[ticker_mask, 'Norm_Adj_Low']
        close = stock_data.loc[ticker_mask, 'Norm_Adj_Close']

        tr1 = np.abs(high - low)
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        true_range = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3})
        true_range['TR'] = true_range.max(axis=1)
        atr = true_range['TR'].rolling(period).mean()

        stock_data.loc[ticker_mask, tr_attribute_name] = true_range['TR']
        stock_data.loc[ticker_mask, atr_attribute_name] = atr

    return stock_data


def calculate_adx(stock_data, period=14):
    """
    Calculates the Average Directional Index (ADX) using Pandas.

    Args:
        stock_data (pandas.DataFrame): The input DataFrame with columns for high, low, and close prices.
        period (int): The number of periods to use for calculating the ADX. Default is 14.

    Returns:
        pandas.Series: A new series containing the ADX values for each row in the input DataFrame.
    """
    adx_attribute_name = str(period) + '_day_adx'

    for ticker in list_stocks(stock_data):
        ticker_mask = stock_data['Stock'] == ticker

        high = stock_data.loc[ticker_mask, 'Norm_Adj_High']
        low = stock_data.loc[ticker_mask, 'Norm_Adj_Low']
        close = stock_data.loc[ticker_mask, 'Norm_Adj_Close']

        # create datafram to hold new stock_data
        dataframe = pd.DataFrame()

        # calculate the True Range (TR)
        dataframe['TR'] = np.nan
        dataframe['TR'] = np.maximum(high - low,
                          np.maximum(abs(high - close.shift()),
                                     abs(low - close.shift())))

        # calculate the Directional Movement (+DM and -DM)
        dataframe['+DM'] = np.where((high - high.shift()) > (low.shift() - low),
                                    np.maximum(high - high.shift(), 0), 0)

        dataframe['-DM'] = np.where((low.shift() - low) > (high - high.shift()),
                                    np.maximum(low.shift() - low, 0), 0)

        # calculate the Directional Indicator (+DI and -DI)
        dataframe['+DI'] = 100 * (dataframe['+DM'].rolling(window=period).sum() /
                                dataframe['TR'].rolling(window=period).sum())

        dataframe['-DI'] = 100 * (dataframe['-DM'].rolling(window=period).sum() /
                                dataframe['TR'].rolling(window=period).sum())

        # calculate the Average Directional Index (ADX)
        dataframe['DX'] = 100 * (abs(dataframe['+DI'] - dataframe['-DI']
                                    ) / (dataframe['+DI'] + dataframe['-DI']))

        dataframe['ADX'] = dataframe['DX'].rolling(window=period).mean()

        stock_data.loc[ticker_mask, adx_attribute_name] = dataframe['ADX']

    # return the ADX values
    return stock_data

def calculate_adl(stock_data):
    """
    The Accumulation/Distribution Indicator is a volume-based technical 
    indicator which uses the relationship between the stock’s price and volume 
    flow to determine the underlying trend of a stock, up, down, or sideways 
    trend of a stock
    
    Args:
        stock_data (pandas.DataFrame): The input DataFrame with columns for high, low, and close prices.

    Returns:
        pandas.Series: A new series containing the ADX values for each row in the input DataFrame.
    """
    print("calculate_adl(stock_data):")
    
    adl_attribute_name = 'adl_value'
    adl_signal_attribute_name = 'adl_signal'

    for ticker in list_stocks(stock_data):
        ticker_mask = stock_data['Stock'] == ticker

        # Use the AccumulationDistributionLine function in the Trading Technical Indicators (tti) library
        adjusted_stock_data = pd.DataFrame()
        adjusted_stock_data["Open"] = stock_data.loc[ticker_mask, "Adj_Open"]
        adjusted_stock_data["High"] = stock_data.loc[ticker_mask, "Adj_High"]
        adjusted_stock_data["Low"] = stock_data.loc[ticker_mask, "Adj_Low"]
        adjusted_stock_data["Close"] = stock_data.loc[ticker_mask, "Adj Close"]
        adjusted_stock_data["Volume"] = stock_data.loc[ticker_mask, "Adj_Volume"]

        # Calculate Accumulation/Distribution Indicator
        ticker_adl = tti.indicators.AccumulationDistributionLine(
            input_data=adjusted_stock_data)
        
        # Generate trading signal
        simulation_data, simulation_statistics, simulation_graph = \
            ticker_adl.getTiSimulation(
                close_values=adjusted_stock_data[['close']], max_exposure=None,
                short_exposure_factor=1.5)
        simulation_statistics.clear()
        simulation_graph.close()

        # Generate signal code
        simulation_data['signal_code'] = simulation_data['signal'].map({'buy': -1, 'sell': 1, 'hold': 0})
       
        stock_data.loc[ticker_mask, adl_attribute_name] = ticker_adl.getTiData()['adl']
        stock_data.loc[ticker_mask, adl_signal_attribute_name] = simulation_data['signal_code']

    return stock_data

def calculate_cmf(stock_data, period=5):
    """
    Chaikin Money Flow measures the amount of Money Flow Volume over a specific period.
    
    Args:
        stock_data (pandas.DataFrame): The input DataFrame with columns for high, low, and close prices.
        period (int): The number of periods to use for calculating the CMF. Default is 5.

    Returns:
        pandas.Series: A new series containing the CMF values for each row in the input DataFrame.
    """
    print("calculate_cmf(", period, "):")

    cmf_attribute_name = 'cmf_value'
    cmf_signal_attribute_name = 'cmf_signal'

    for ticker in list_stocks(stock_data):
        ticker_mask = stock_data['Stock'] == ticker

        # Use the ChaikinMoneyFlow function in the Trading Technical Indicators (tti) library
        adjusted_stock_data = pd.DataFrame()
        adjusted_stock_data["Open"] = stock_data.loc[ticker_mask, "Adj_Open"]
        adjusted_stock_data["High"] = stock_data.loc[ticker_mask, "Adj_High"]
        adjusted_stock_data["Low"] = stock_data.loc[ticker_mask, "Adj_Low"]
        adjusted_stock_data["Close"] = stock_data.loc[ticker_mask, "Adj Close"]
        adjusted_stock_data["Volume"] = stock_data.loc[ticker_mask, "Adj_Volume"]

        # Calculate ChaikinMoneyFlow
        ticker_cmf = tti.indicators.ChaikinMoneyFlow(
            input_data=adjusted_stock_data, period=period)

        # Generate trading signal
        simulation_data, simulation_statistics, simulation_graph = \
            ticker_cmf.getTiSimulation(
                close_values=adjusted_stock_data[['close']], max_exposure=None,
                short_exposure_factor=1.5)
        simulation_statistics.clear()
        simulation_graph.close()
        
        # Generate signal code
        simulation_data['signal_code'] = simulation_data['signal'].map(
            {'buy': -1, 'sell': 1, 'hold': 0})

        stock_data.loc[ticker_mask, cmf_attribute_name] = ticker_cmf.getTiData()['cmf']
        stock_data.loc[ticker_mask, cmf_signal_attribute_name] = simulation_data['signal_code']

    return stock_data

def calculate_co(stock_data):
    """
    The oscillator measures the accumulation-distribution line of moving 
    average convergence-divergence (MACD).    
    
    Args:
        stock_data (pandas.DataFrame): The input DataFrame with columns for high, low, and close prices.

    Returns:
        pandas.Series: A new series containing the values for each row in the input DataFrame.
    """
    print("calculate_co(stock_data):")

    co_attribute_name = 'co_value'
    co_signal_attribute_name = 'co_signal'

    for ticker in list_stocks(stock_data):
        ticker_mask = stock_data['Stock'] == ticker

        # Use the ChaikinOscillator function in the Trading Technical Indicators (tti) library
        adjusted_stock_data = pd.DataFrame()
        adjusted_stock_data["Open"] = stock_data.loc[ticker_mask, "Adj_Open"]
        adjusted_stock_data["High"] = stock_data.loc[ticker_mask, "Adj_High"]
        adjusted_stock_data["Low"] = stock_data.loc[ticker_mask, "Adj_Low"]
        adjusted_stock_data["Close"] = stock_data.loc[ticker_mask, "Adj Close"]
        adjusted_stock_data["Volume"] = stock_data.loc[ticker_mask, "Adj_Volume"]

        # Calculate Indicator
        ticker_co = tti.indicators.ChaikinOscillator(
            input_data=adjusted_stock_data)

        # Generate trading signal
        simulation_data, simulation_statistics, simulation_graph = \
            ticker_co.getTiSimulation(
                close_values=adjusted_stock_data[['close']], max_exposure=None,
                short_exposure_factor=1.5)
        simulation_statistics.clear()
        simulation_graph.close()

        # Generate signal code
        simulation_data['signal_code'] = simulation_data['signal'].map(
            {'buy': -1, 'sell': 1, 'hold': 0})

        stock_data.loc[ticker_mask, co_attribute_name] = ticker_co.getTiData()[
            'co']
        stock_data.loc[ticker_mask,
                       co_signal_attribute_name] = simulation_data['signal_code']

    return stock_data


def calculate_cmo(stock_data, period=5):
    """
    The Chande Momentum Oscillator (CMO) is a technical momentum indicator 
    developed by Tushar Chande. The CMO indicator is created by calculating 
    the difference between the sum of all recent higher closes and the sum of 
    all recent lower closes and then dividing the result by the sum of all 
    price movement over a given time period.
    
    Args:
        stock_data (pandas.DataFrame): The input DataFrame with columns for high, low, and close prices.
        period (int): The number of periods to use for calculating the CMF. Default is 5.

    Returns:
        pandas.Series: A new series containing the values for each row in the input DataFrame.
    """
    print("calculate_cmo(", period, "):")

    cmo_attribute_name = 'cmo_value'
    cmo_signal_attribute_name = 'cmo_signal'

    for ticker in list_stocks(stock_data):
        ticker_mask = stock_data['Stock'] == ticker

        # Use the ChandeMomentumOscillator function in the Trading Technical Indicators (tti) library
        adjusted_stock_data = pd.DataFrame()
        adjusted_stock_data["Open"] = stock_data.loc[ticker_mask, "Adj_Open"]
        adjusted_stock_data["High"] = stock_data.loc[ticker_mask, "Adj_High"]
        adjusted_stock_data["Low"] = stock_data.loc[ticker_mask, "Adj_Low"]
        adjusted_stock_data["Close"] = stock_data.loc[ticker_mask, "Adj Close"]
        adjusted_stock_data["Volume"] = stock_data.loc[ticker_mask, "Adj_Volume"]

        # Calculate ChandeMomentumOscillator
        ticker_cmf = tti.indicators.ChandeMomentumOscillator(
            input_data=adjusted_stock_data, period=period)

        # Generate trading signal
        simulation_data, simulation_statistics, simulation_graph = \
            ticker_cmf.getTiSimulation(
                close_values=adjusted_stock_data[['close']], max_exposure=None,
                short_exposure_factor=1.5)
        simulation_statistics.clear()
        simulation_graph.close()

        # Generate signal code
        simulation_data['signal_code'] = simulation_data['signal'].map(
            {'buy': -1, 'sell': 1, 'hold': 0})

        stock_data.loc[ticker_mask, cmo_attribute_name] = ticker_cmf.getTiData()[
            'cmo']
        stock_data.loc[ticker_mask,
                       cmo_signal_attribute_name] = simulation_data['signal_code']

    return stock_data


def calculate_cci(stock_data, period=5):
    """
    The Commodity Channel Index (CCI) is a technical indicator that measures 
    the difference between the current price and the historical average price. 
    When the CCI is above zero, it indicates the price is above the historic 
    average. Conversely, when the CCI is below zero, the price is below the 
    historic average.
        
    Args:
        stock_data (pandas.DataFrame): The input DataFrame with columns for high, low, and close prices.
        period (int): The number of periods to use for calculating the CMF. Default is 5.

    Returns:
        pandas.Series: A new series containing the values for each row in the input DataFrame.
    """
    print("calculate_cci(", period, "):")

    cci_attribute_name = 'cci_value'
    cci_signal_attribute_name = 'cci_signal'

    for ticker in list_stocks(stock_data):
        ticker_mask = stock_data['Stock'] == ticker

        # Use the CommodityChannelIndex function in the Trading Technical Indicators (tti) library
        adjusted_stock_data = pd.DataFrame()
        adjusted_stock_data["Open"] = stock_data.loc[ticker_mask, "Adj_Open"]
        adjusted_stock_data["High"] = stock_data.loc[ticker_mask, "Adj_High"]
        adjusted_stock_data["Low"] = stock_data.loc[ticker_mask, "Adj_Low"]
        adjusted_stock_data["Close"] = stock_data.loc[ticker_mask, "Adj Close"]
        adjusted_stock_data["Volume"] = stock_data.loc[ticker_mask, "Adj_Volume"]

        # Calculate CommodityChannelIndex
        ticker_cci = tti.indicators.CommodityChannelIndex(
            input_data=adjusted_stock_data, period=period)

        # Generate trading signal
        simulation_data, simulation_statistics, simulation_graph = \
            ticker_cci.getTiSimulation(
                close_values=adjusted_stock_data[['close']], max_exposure=None,
                short_exposure_factor=1.5)
        simulation_statistics.clear()
        simulation_graph.close()

        # Generate signal code
        simulation_data['signal_code'] = simulation_data['signal'].map(
            {'buy': -1, 'sell': 1, 'hold': 0})

        stock_data.loc[ticker_mask, cci_attribute_name] = ticker_cci.getTiData()[
            'cci']
        stock_data.loc[ticker_mask,
                       cci_signal_attribute_name] = simulation_data['signal_code']

    return stock_data


def calculate_dpo(stock_data, period=6):
    """
    A detrended price oscillator, used in technical analysis, strips out price 
    trends in an effort to estimate the length of price cycles from peak to 
    peak or trough to trough.

    Unlike other oscillators, such as the stochastic or moving average 
    convergence divergence (MACD), the DPO is not a momentum indicator. It 
    instead highlights peaks and troughs in price, which are used to estimate 
    buy and sell points in line with the historical cycle.
        
    Args:
        stock_data (pandas.DataFrame): The input DataFrame with columns for high, low, and close prices.
        period (int): The number of periods to use for calculating the CMF. Default is 6.

    Returns:
        pandas.Series: A new series containing the values for each row in the input DataFrame.
    """
    print("calculate_dpo(", period, "):")

    dpo_attribute_name = 'dpo_value'
    dpo_signal_attribute_name = 'dpo_signal'

    for ticker in list_stocks(stock_data):
        ticker_mask = stock_data['Stock'] == ticker

        # Use the DetrendedPriceOscillator function in the Trading Technical Indicators (tti) library
        adjusted_stock_data = pd.DataFrame()
        adjusted_stock_data["Open"] = stock_data.loc[ticker_mask, "Adj_Open"]
        adjusted_stock_data["High"] = stock_data.loc[ticker_mask, "Adj_High"]
        adjusted_stock_data["Low"] = stock_data.loc[ticker_mask, "Adj_Low"]
        adjusted_stock_data["Close"] = stock_data.loc[ticker_mask, "Adj Close"]
        adjusted_stock_data["Volume"] = stock_data.loc[ticker_mask, "Adj_Volume"]

        # Calculate DetrendedPriceOscillator
        ticker_dpo = tti.indicators.DetrendedPriceOscillator(
            input_data=adjusted_stock_data, period=period)

        # Generate trading signal
        simulation_data, simulation_statistics, simulation_graph = \
            ticker_dpo.getTiSimulation(
                close_values=adjusted_stock_data[['close']], max_exposure=None,
                short_exposure_factor=1.5)
        simulation_statistics.clear()
        simulation_graph.close()

        # Generate signal code
        simulation_data['signal_code'] = simulation_data['signal'].map(
            {'buy': -1, 'sell': 1, 'hold': 0})

        stock_data.loc[ticker_mask, dpo_attribute_name] = ticker_dpo.getTiData()[
            'dpo']
        stock_data.loc[ticker_mask,
                       dpo_signal_attribute_name] = simulation_data['signal_code']

    return stock_data


def calculate_dmi(stock_data):
    """
    The directional movement index (DMI) is an indicator developed by J. Welles 
    Wilder in 1978 that identifies in which direction the price of an asset is 
    moving. The indicator does this by comparing prior highs and lows and 
    drawing two lines: a positive directional movement line (+DI) and a 
    negative directional movement line (-DI).
    
    Args:
        stock_data (pandas.DataFrame): The input DataFrame with columns for high, low, and close prices.

    Returns:
        pandas.Series: A new series containing the values for each row in the input DataFrame.
    """
    print("calculate_dmi():")

    pdi_attribute_name = 'dmi_pdi'
    mdi_attribute_name = 'dmi_mdi'
    dx_attribute_name = 'dmi_dx'
    adx_attribute_name = 'dmi_adx'
    adxr_attribute_name = 'dmi_adxr'
    dmi_signal_attribute_name = 'dmi_signal'

    for ticker in list_stocks(stock_data):
        ticker_mask = stock_data['Stock'] == ticker

        # Use the DirectionalMovementIndex function in the Trading Technical Indicators (tti) library
        adjusted_stock_data = pd.DataFrame()
        adjusted_stock_data["Open"] = stock_data.loc[ticker_mask, "Adj_Open"]
        adjusted_stock_data["High"] = stock_data.loc[ticker_mask, "Adj_High"]
        adjusted_stock_data["Low"] = stock_data.loc[ticker_mask, "Adj_Low"]
        adjusted_stock_data["Close"] = stock_data.loc[ticker_mask, "Adj Close"]
        adjusted_stock_data["Volume"] = stock_data.loc[ticker_mask, "Adj_Volume"]

        # Calculate Indicator
        ticker_dmi = tti.indicators.DirectionalMovementIndex(
            input_data=adjusted_stock_data)

        # Generate trading signal
        simulation_data, simulation_statistics, simulation_graph = \
            ticker_dmi.getTiSimulation(
                close_values=adjusted_stock_data[['close']], max_exposure=None,
                short_exposure_factor=1.5)
        simulation_statistics.clear()
        simulation_graph.close()

        # Generate signal code
        simulation_data['signal_code'] = simulation_data['signal'].map(
            {'buy': -1, 'sell': 1, 'hold': 0})

        stock_data.loc[ticker_mask, pdi_attribute_name] = ticker_dmi.getTiData()['+di']
        stock_data.loc[ticker_mask, mdi_attribute_name] = ticker_dmi.getTiData()['-di']
        stock_data.loc[ticker_mask, dx_attribute_name] = ticker_dmi.getTiData()['dx']
        stock_data.loc[ticker_mask, adx_attribute_name] = ticker_dmi.getTiData()['adx']
        stock_data.loc[ticker_mask, adxr_attribute_name] = ticker_dmi.getTiData()['adxr']
        stock_data.loc[ticker_mask, dmi_signal_attribute_name] = simulation_data['signal_code']

    return stock_data


def calculate_dema(stock_data, period=5):
    """
    The double exponential moving average (DEMA) is a variation on a technical 
    indicator used to identify a potential uptrend or downtrend in the price of 
    a stock or other asset. A moving average tracks the average price of an 
    asset over a period to spot the point at which it establishes a new trend, 
    moving above or below its average price.
    
    Args:
        stock_data (pandas.DataFrame): The input DataFrame with columns for high, low, and close prices.
        period (int): The number of periods to use for calculating the CMF. Default is 6.

    Returns:
        pandas.Series: A new series containing the values for each row in the input DataFrame.
    """
    print("calculate_dema(", period, "):")

    dema_attribute_name = 'dema_value'
    dema_signal_attribute_name = 'dema_signal'

    for ticker in list_stocks(stock_data):
        ticker_mask = stock_data['Stock'] == ticker

        # Use the DoubleExponentialMovingAverage function in the Trading Technical Indicators (tti) library
        adjusted_stock_data = pd.DataFrame()
        adjusted_stock_data["Open"] = stock_data.loc[ticker_mask, "Adj_Open"]
        adjusted_stock_data["High"] = stock_data.loc[ticker_mask, "Adj_High"]
        adjusted_stock_data["Low"] = stock_data.loc[ticker_mask, "Adj_Low"]
        adjusted_stock_data["Close"] = stock_data.loc[ticker_mask, "Adj Close"]
        adjusted_stock_data["Volume"] = stock_data.loc[ticker_mask, "Adj_Volume"]

        # Calculate DoubleExponentialMovingAverage
        ticker_dema = tti.indicators.DoubleExponentialMovingAverage(
            input_data=adjusted_stock_data, period=period)

        # Generate trading signal
        simulation_data, simulation_statistics, simulation_graph = \
            ticker_dema.getTiSimulation(
                close_values=adjusted_stock_data[['close']], max_exposure=None,
                short_exposure_factor=1.5)
        simulation_statistics.clear()
        simulation_graph.close()

        # Generate signal code
        simulation_data['signal_code'] = simulation_data['signal'].map(
            {'buy': -1, 'sell': 1, 'hold': 0})

        stock_data.loc[ticker_mask, dema_attribute_name] = ticker_dema.getTiData()[
            'dema']
        stock_data.loc[ticker_mask,
                       dema_signal_attribute_name] = simulation_data['signal_code']

    return stock_data


# def calculate_eom(stock_data, period=40):
#     """
#     Ease of Movement (EOM or EMV) indicator is a technical study that attempts 
#     to quantify a mix of momentum and volume information into one value.  The 
#     intent is to use this value to discern whether prices are able to rise, or 
#     fall, with little resistance in the directional movement.

#     Theoretically, if prices move easily, they will continue to do so for a 
#     period of time that can be traded effectively.
    
#     Args:
#         stock_data (pandas.DataFrame): The input DataFrame with columns for high, low, and close prices.
#         period (int): The number of periods to use for calculating the CMF. Default is 6.

#     Returns:
#         pandas.Series: A new series containing the values for each row in the input DataFrame.
#     """
#     print("calculate_eom(", period, "):")

#     emv_attribute_name = 'emv_value'
#     emv_ma_attribute_name = 'emv_ma_value'
#     emv_signal_attribute_name = 'emv_signal'

#     for ticker in list_stocks(stock_data):
#         ticker_mask = stock_data['Stock'] == ticker

#         # Use the EaseOfMovement function in the Trading Technical Indicators (tti) library
#         adjusted_stock_data = pd.DataFrame()
#         adjusted_stock_data["Open"] = stock_data.loc[ticker_mask, "Adj_Open"]
#         adjusted_stock_data["High"] = stock_data.loc[ticker_mask, "Adj_High"]
#         adjusted_stock_data["Low"] = stock_data.loc[ticker_mask, "Adj_Low"]
#         adjusted_stock_data["Close"] = stock_data.loc[ticker_mask, "Adj Close"]
#         adjusted_stock_data["Volume"] = stock_data.loc[ticker_mask, "Adj_Volume"]

#         # Calculate EaseOfMovement
#         ticker_dema = tti.indicators.EaseOfMovement(
#             input_data=adjusted_stock_data, period=period)

#         # Generate trading signal
#         simulation_data, simulation_statistics, simulation_graph = \
#             ticker_dema.getTiSimulation(
#                 close_values=adjusted_stock_data[['close']], max_exposure=None,
#                 short_exposure_factor=1.5)
#         simulation_statistics.clear()
#         simulation_graph.close()

#         # Generate signal code
#         simulation_data['signal_code'] = simulation_data['signal'].map(
#             {'buy': -1, 'sell': 1, 'hold': 0})

#         stock_data.loc[ticker_mask, emv_attribute_name] = ticker_dema.getTiData()[
#             'emv']
#         stock_data.loc[ticker_mask, emv_ma_attribute_name] = ticker_dema.getTiData()[
#             'emv_ma']
#         stock_data.loc[ticker_mask,
#                        emv_signal_attribute_name] = simulation_data['signal_code']

#     return stock_data


def calculate_tti(stock_data, tti_function, period=None):
    """
    Calculate the technical indicator
        
    Args:
        stock_data (pandas.DataFrame): The input DataFrame with columns for high, low, and close prices.
        tti_function (string): Trading Technical Indicator function name.
        period (int): The number of periods to use for calculating the CMF. Default is 6.

    Returns:
        pandas.Series: A new series containing the values for each row in the input DataFrame.
    """
    print("calculate_tti(", tti_function, ":", period, "):")

    for ticker in list_stocks(stock_data):
        ticker_mask = stock_data['Stock'] == ticker

        # Use the tii function in the Trading Technical Indicators (tti) library
        adjusted_stock_data = pd.DataFrame()
        adjusted_stock_data["Open"] = stock_data.loc[ticker_mask, "Adj_Open"]
        adjusted_stock_data["High"] = stock_data.loc[ticker_mask, "Adj_High"]
        adjusted_stock_data["Low"] = stock_data.loc[ticker_mask, "Adj_Low"]
        adjusted_stock_data["Close"] = stock_data.loc[ticker_mask, "Adj Close"]
        adjusted_stock_data["Volume"] = stock_data.loc[ticker_mask, "Adj_Volume"]

        # Calculate technical indcator
        if (period is None):
            ticker_value = call_lib_function("tti.indicators", tti_function,
                                             input_data=adjusted_stock_data)
        else:
            ticker_value = call_lib_function("tti.indicators", tti_function,
                      input_data=adjusted_stock_data, period=period)
        
        # Generate trading signal
        simulation_data, simulation_statistics, simulation_graph = \
            ticker_value.getTiSimulation(
                close_values=adjusted_stock_data[['close']], max_exposure=None,
                short_exposure_factor=1.5)
        simulation_statistics.clear()
        simulation_graph.close()

        # Generate signal code
        simulation_data['signal_code'] = simulation_data['signal'].map(
            {'buy': -1, 'sell': 1, 'hold': 0})

        print('\nticker_value.getTiData()\n', ticker_value.getTiData())
        
        # stock_data.loc[ticker_mask, tti_function+"."+tti_col1_name+
        #                '.value'] = ticker_value.getTiData()[tti_col1_name]
        
        # if (tti_col2_name is not None):
        #     stock_data.loc[ticker_mask, tti_function+'.'+tti_col2_name+'.value'] = ticker_value.getTiData()[tti_col2_name]
    
        # for col_name in (tti_col_name):
        #     stock_data.loc[ticker_mask, tti_function+'.'+col_name +
        #                 '.value'] = ticker_value.getTiData()[col_name]

        df = ticker_value.getTiData()
        
        for col_name in df.columns:
            stock_data.loc[ticker_mask, tti_function+'.'+col_name +
                           '.value'] = ticker_value.getTiData()[col_name]

        stock_data.loc[ticker_mask, tti_function+'.' +
                       'signal'] = simulation_data['signal_code']

        print('\nticker_value.getTiData()\n', df)
        print('\nstock_data columns\n', stock_data.columns)
        print('\nstock_data\n', stock_data)

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
    stock_data = calculate_adl(stock_data)
    stock_data = calculate_co(stock_data)
    stock_data = calculate_dmi(stock_data)
    stock_data = calculate_tti(stock_data, "FibonacciRetracement")

    # def calculate_tti(stock_data, tti_function, period=None, tti_col1_name=None, tti_col2_name=None):

    ticker_fields = ['Norm_Adj_Open',
                     'Norm_Adj_High',
                     'Norm_Adj_Low',
                     'Norm_Adj_Close',
                     'Norm_Adj_Volume',
                     'obv']
    periods = list(range(3, 31)) + [60, 90, 180, 300]
    
    for period in periods:
        stock_data = calculate_aggregate(stock_data, period)
        
        for field in ticker_fields:
            attribute_name = str(period) + '_day_' + field
            stock_data = calculate_sma(stock_data, attribute_name + '_sma', field, period)
            stock_data = calculate_ema(stock_data, attribute_name + '_ema', field, period)
            stock_data = calculate_bb(stock_data, attribute_name + '_boiler_band', window_size=period)

    for period in range(3, 41):
        stock_data = stock_data.copy()
        attribute_name = str(period) + '_day_'
        stock_data = calculate_rsi(stock_data, attribute_name + 'rsi', period)
        stock_data = stochastic_oscillator(stock_data, attribute_name, period)
        stock_data = calculate_stoch_rsi(stock_data, period)
        stock_data = calculate_atr(stock_data, period)
        stock_data = calculate_adx(stock_data, period)
        stock_data = calculate_cmf(stock_data, period)
        stock_data = calculate_cmo(stock_data, period)
        stock_data = calculate_cci(stock_data, period)
        stock_data = calculate_dpo(stock_data, period)
        stock_data = calculate_dema(stock_data, period)
        stock_data = calculate_tti(stock_data, "EaseOfMovement", period)
        stock_data = calculate_tti(stock_data, "Envelopes", period)

    stock_data = calculate_macd(stock_data)
    stock_data = calculate_rps(stock_data, 'FXAIX')

    return stock_data
