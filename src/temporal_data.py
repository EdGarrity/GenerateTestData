"""
Add temporal data to the stock data.
"""
import pandas as pd

def add_trading_days_in_year(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the number of trading days in each year for each stock in the DataFrame.

    This function calculates the number of trading days in each year for each stock
    in the provided DataFrame based on the non-null values in the 'Adj Close' column.

    Parameters:
        stock_data (pd.DataFrame): A DataFrame containing stock data with the following columns:
            - 'Symbol': The stock symbol or identifier for each data record.
            - 'Date': The date for each data record. Should be set as the DataFrame index.
            - 'Adj Close': The adjusted closing price of the stock for each data record.

    Returns:
        pd.DataFrame: A new DataFrame with the 'TradingDaysInYear' field added.
            The returned DataFrame will be the same as the input 'stock_data'
            but with an additional column named 'TradingDaysInYear'.
            The 'TradingDaysInYear' column will contain the number of trading days in each year
            for the respective stock and year combination. For non-trading days or missing data,
            the 'TradingDaysInYear' value will be 0.

    Note:
        The 'Date' column should be set as the DataFrame index, and the data should be sorted in
        chronological order based on the date for correct results.

    Example:
        >>> stock_data = pd.DataFrame({
        ...     'Symbol': ['AAPL', 'AAPL', 'AAPL', 'GOOGL', 'GOOGL', 'GOOGL'],
        ...     'Date': ['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-03', '2022-01-04', '2022-01-06'],
        ...     'Adj Close': [180.05, 182.21, None, 3100.0, 3095.5, 3134.0]
        ... })
        >>> stock_data.set_index('Date', inplace=True)
        >>> result = add_trading_days_in_year(stock_data)
        >>> print(result)
                     Symbol  Adj Close  TradingDaysInYear
        Date
        2022-01-03    AAPL     180.05                 1
        2022-01-04    AAPL     182.21                 1
        2022-01-05    AAPL       NaN                 1
        2022-01-03   GOOGL    3100.00                 2
        2022-01-04   GOOGL    3095.50                 2
        2022-01-06   GOOGL    3134.00                 2
    """
    # Group data by stock symbol and year
    grouped_data = stock_data.groupby([stock_data.index.year, 'Symbol'], group_keys=False)

    # Calculate TradingDaysInYear for each stock and year
    stock_data['TradingDaysInYear'] = grouped_data['Adj Close'].transform(lambda x: x.notnull().sum())

    return stock_data

def add_trading_days_since_start_of_year(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds temporal data to the input DataFrame.

    The input stock_data DataFrame consists of stock information for several stocks from the first trading day of 1947 to the last trading day of 2022.  The DataFrame comtains many columns.  The name of the stock is in the 'Stock' column of the DataFrame.  The date is the row.name of the DataFrame

    For each record in the DataFrame, this function will add the following field:

    TradingDaysSinceStartOfYear
        Where TradingDaysSinceStartOfYear is calculated by counting how many days the DataFrame has information for that stock in the given year, not how many calendar days are in the year. (1 = First trading day of the year, 2 = second trading day of the year, ...)

    Parameters:
        stock_data (pd.DataFrame): The DataFrame containing stock market or financial data. The structure of the DataFrame is as follows:
        'Stock': A string that represents the stock trading symbol
        'Date': Column representing dates in a valid date format, compatible with the 'pd.to_datetime()' function

    Returns:
        pd.DataFrame: A DataFrame with the updated 'stock_data' including the added temporal-related information.

    Example:

        Suppose we have a DataFrame 'stock_data' with columns 'Stock' and 'Date', and we want to add temporal data to it. We can use the function as follows:

        >>> updated_data = add_temporal_data(stock_data)
        >>> print(updated_data)

    Note:
        - The 'Date' column in the 'stock_data' DataFrame should be in a valid date format, compatible with the 'pd.to_datetime()' function used in this function.
        - The 'stock_data' DataFrame will be modified in place to include the additional temporal data.
        
    Test Example:
        stock_data = pd.DataFrame({
            'Symbol':    ['AAPL',       'AAPL',       'AAPL',       'AAPL',       'AAPL',       'GOOGL',      'GOOGL',      'GOOGL',      'GOOGL',      'GOOGL'],
            'Date':      ['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-02', '2022-01-03'],
            'Adj Close': [ 180.05,       182.21,       None,         182.22,       182.21,       3100.0,       3095.5,       3134.0,       3144.00,      3144.00]
        })
        stock_data.set_index('Date', inplace=True)
        result = add_trading_days_since_start_of_year(stock_data)
        print(result)

    Test Output:
            Symbol  Adj Close  TradingDaysSinceStartOfYear
        Date
        2022-01-03   AAPL     180.05                            1
        2022-01-04   AAPL     182.21                            2
        2022-01-05   AAPL        NaN                            2
        2022-01-06   AAPL     182.22                            3
        2022-01-07   AAPL     182.21                            4
        2022-01-03  GOOGL    3100.00                            1
        2022-01-04  GOOGL    3095.50                            2
        2022-01-05  GOOGL    3134.00                            3
        2023-01-02  GOOGL    3144.00                            1
        2023-01-03  GOOGL    3144.00                            2
    """
    # Step 1: Extract the year from the dates
    stock_data['Year'] = pd.to_datetime(stock_data.index).year

    # Step 2: Group by 'Symbol' and 'Year' and calculate the number of trading days since the start of the year
    stock_data['TradingDaysSinceStartOfYear'] = stock_data.groupby(['Symbol', 'Year'], group_keys=False)['Adj Close'].apply(lambda x: x.notnull().cumsum())

    # Drop the 'Year' column as it is no longer needed
    stock_data.drop(columns=['Year'], inplace=True)

    return stock_data

def add_trading_days_left_in_year(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds temporal data to the input DataFrame.

    The input stock_data DataFrame consists of stock information for several stocks from the first trading day of 1947 to the last trading day of 2022.  The DataFrame comtains many columns.  The name of the stock is in the 'Stock' column of the DataFrame.  The date is the row.name of the DataFrame

    For each record in the DataFrame, this function will add the following field:

    TradingDaysLeftInYear
        The TradingDaysLeftInYear is calculated as TradingDaysInYear - TradingDaysSinceStartOfYear

    Parameters:
        stock_data (pd.DataFrame): The DataFrame containing stock market or financial data. The structure of the DataFrame is as follows:
        'Stock': A string that represents the stock trading symbol
        'Date': Column representing dates in a valid date format, compatible with the 'pd.to_datetime()' function

    Returns:
        pd.DataFrame: A DataFrame with the updated 'stock_data' including the added temporal-related information.

    Example:

        Suppose we have a DataFrame 'stock_data' with columns 'Stock' and 'Date', and we want to add temporal data to it. We can use the function as follows:

        >>> stock_data = add_trading_days_in_year(stock_data)
        >>> stock_data = add_trading_days_since_start_of_year(stock_data)
        >>> stock_data = add_trading_days_left_in_year(stock_data)
        >>> print(stock_data)

    Note:
        - The 'Date' column in the 'stock_data' DataFrame should be in a valid date format, compatible with the 'pd.to_datetime()' function used in this function.
        - The 'stock_data' DataFrame will be modified in place to include the additional temporal data.
        
    """
    # Calculate TradingDaysLeftInYear as TradingDaysInYear - TradingDaysSinceStartOfYear
    stock_data['TradingDaysLeftInYear'] = stock_data['TradingDaysInYear'] - stock_data['TradingDaysSinceStartOfYear']

    return stock_data

def add_trading_days_in_week(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the number of trading days in each week for each stock in the DataFrame.

    This function calculates the number of trading days in each week for each stock
    in the provided DataFrame based on the non-null values in the 'Adj Close' column.

    Parameters:
        stock_data (pd.DataFrame): A DataFrame containing stock data with the following columns:
            - 'Symbol': The stock symbol or identifier for each data record.
            - 'Date': The date for each data record. Should be set as the DataFrame index.
            - 'Adj Close': The adjusted closing price of the stock for each data record.

    Returns:
        pd.DataFrame: A new DataFrame with the 'TradingDaysInWeek' field added.
            The returned DataFrame will be the same as the input 'stock_data'
            but with an additional column named 'TradingDaysInWeek'.
            The 'TradingDaysInWeek' column will contain the number of trading days in each week
            for the respective stock, week, and year combination. For non-trading days or missing data,
            the 'TradingDaysInWeek' value will be 0.

    Note:
        The 'Date' column should be set as the DataFrame index, and the data should be sorted in
        chronological order based on the date for correct results.

    Example:
        >>> stock_data = pd.DataFrame({
        ...     'Symbol':    ['AAPL',       'AAPL',       'AAPL',       'GOOGL',      'GOOGL',      'GOOGL',      'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL'],
        ...     'Date':      ['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-03', '2022-01-04', '2022-01-06', 2023-01-17, 2023-01-18, 2023-01-19, 2023-02-21, 2023-02-22, 2023-02-23],
        ...     'Adj Close': [180.05,       182.21,       None,         3100.0,       3095.5,       3134.0,       180.00,     181.00,     182.00,     183.00,     184.00,     185.00]
        ... })
        >>> stock_data.set_index('Date', inplace=True)
        >>> result = add_trading_days_in_year(stock_data)
        >>> print(result)
                     Symbol  Adj Close  TradingDaysInWeek
        Date
        2022-01-03    AAPL     180.05                 5
        2022-01-04    AAPL     182.21                 5
        2022-01-05    AAPL       NaN                  5
        2022-01-03   GOOGL    3100.00                 5
        2022-01-04   GOOGL    3095.50                 5
        2022-01-06   GOOGL    3134.00                 5
        2023-01-17    AAPL     180.00                 4
        2023-01-18    AAPL     181.00                 4
        2023-01-19    AAPL     182.00                 4
        2023-02-21    AAPL     183.00                 4
        2023-02-22    AAPL     184.00                 4
        2023-02-23    AAPL     185.00                 4
    """
    # Convert the 'Date' column to a DatetimeIndex
    stock_data.index = pd.to_datetime(stock_data.index)

    # Group data by stock symbol, year, and week
    # FutureWarning: weekofyear and week have been deprecated, please use DatetimeIndex.isocalendar().week 
    # instead, which returns a Series. To exactly reproduce the behavior of week and weekofyear and return 
    # an Index, you may call pd.Int64Index(idx.isocalendar().week)
    # grouped_data = stock_data.groupby([stock_data.index.year, stock_data.index.week, 'Symbol'])
    grouped_data = stock_data.groupby([stock_data.index.year, stock_data.index.isocalendar().week, 'Symbol'])

    # Calculate TradingDaysInWeek for each stock, year, and week
    stock_data['TradingDaysInWeek'] = grouped_data['Adj Close'].transform(lambda x: x.notnull().sum())

    return stock_data

def add_day_of_week(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    A numeric value representing the day of the week (0 = Sunday, 1 = Monday, ...) 
    for each stock in the DataFrame.

    Parameters:
        stock_data (pd.DataFrame): A DataFrame containing stock data with the following columns:
            - 'Symbol': The stock symbol or identifier for each data record.
            - 'Date': The date for each data record. Should be set as the DataFrame index.
            - 'Adj Close': The adjusted closing price of the stock for each data record.

    Returns:
        pd.DataFrame: A new DataFrame with the 'DayOfWeek' field added.
            The returned DataFrame will be the same as the input 'stock_data'
            but with an additional column named 'DayOfWeek'.
            The 'DayOfWeek' column will contain the number of trading days in each week
            for the respective stock, week, and year combination. For non-trading days or missing data,
            the 'DayOfWeek' value will be 0.

    Note:
        The 'Date' column should be set as the DataFrame index, and the data should be sorted in
        chronological order based on the date for correct results.

    Example:
        >>> stock_data = pd.DataFrame({
        ...     'Symbol':    ['AAPL',       'AAPL',       'AAPL',       'GOOGL',      'GOOGL',      'GOOGL',      'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL'],
        ...     'Date':      ['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-03', '2022-01-04', '2022-01-06', 2023-01-17, 2023-01-18, 2023-01-19, 2023-02-21, 2023-02-22, 2023-02-23],
        ...     'Adj Close': [180.05,       182.21,       None,         3100.0,       3095.5,       3134.0,       180.00,     181.00,     182.00,     183.00,     184.00,     185.00]
        ... })
        >>> stock_data.set_index('Date', inplace=True)
        >>> result = add_trading_days_in_year(stock_data)
        >>> print(result)
                     Symbol  Adj Close  DayOfWeek
        Date
        2022-01-03    AAPL     180.05   1
        2022-01-04    AAPL     182.21   2
        2022-01-05    AAPL       NaN    3
        2022-01-03   GOOGL    3100.00   1
        2022-01-04   GOOGL    3095.50   2
        2022-01-06   GOOGL    3134.00   4
        2023-01-17    AAPL     180.00   2
        2023-01-18    AAPL     181.00   3
        2023-01-19    AAPL     182.00   4
        2023-02-21    AAPL     183.00   2
        2023-02-22    AAPL     184.00   3
        2023-02-23    AAPL     185.00   4
    """

    # Extract the day of the week (0=Monday, 1=Tuesday, ..., 6=Sunday)
    stock_data['DayOfWeek'] = stock_data.index.dayofweek

    # Update the index name to 'Date' (if needed)
    # stock_data.index.name = 'Date'

    return stock_data

def add_week_of_month(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    A numeric value respresenting the week of the month (1 = first week, 2 = second week, ...) 
    for each stock in the DataFrame.

    Parameters:
        stock_data (pd.DataFrame): A DataFrame containing stock data with the following columns:
            - 'Symbol': The stock symbol or identifier for each data record.
            - 'Date': The date for each data record. Should be set as the DataFrame index.
            - 'Adj Close': The adjusted closing price of the stock for each data record.

    Returns:
        pd.DataFrame: A new DataFrame with the 'WeekOfMonth' field added.
            The returned DataFrame will be the same as the input 'stock_data'
            but with an additional column named 'WeekOfMonth'.
            The 'WeekOfMonth' column will contain the number respresenting the week of the month
            for the respective stock, week, month, and year combination. For non-trading days or 
			missing data, the 'WeekOfMonth' value will be 0.

    Note:
        The 'Date' column should be set as the DataFrame index, and the data should be sorted in
        chronological order based on the date for correct results.

    Example:
        >>> stock_data = pd.DataFrame({
        ...     'Symbol':    ['AAPL',       'AAPL',       'AAPL',       'GOOGL',      'GOOGL',      'GOOGL',      'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL'],
        ...     'Date':      ['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-03', '2022-01-04', '2022-01-06', 2023-01-17, 2023-01-18, 2023-01-19, 2023-02-21, 2023-02-22, 2023-02-23, 2023-02-27, 2023-02-28],
        ...     'Adj Close': [180.05,       182.21,       None,         3100.0,       3095.5,       3134.0,       180.00,     181.00,     182.00,     183.00,     184.00,     185.00,     185.00,     185.00]
        ... })
        >>> stock_data.set_index('Date', inplace=True)
        >>> result = add_trading_days_in_year(stock_data)
        >>> print(result)
                     Symbol  Adj Close  WeekOfMonth
        Date
        2022-01-03    AAPL     180.05   1
        2022-01-04    AAPL     182.21   1
        2022-01-05    AAPL       NaN    0
        2022-01-03   GOOGL    3100.00   1
        2022-01-04   GOOGL    3095.50   1
        2022-01-06   GOOGL    3134.00   1
        2023-01-17    AAPL     180.00   3
        2023-01-18    AAPL     181.00   3
        2023-01-19    AAPL     182.00   3
        2023-02-21    AAPL     183.00   3
        2023-02-22    AAPL     184.00   3
        2023-02-23    AAPL     185.00   3
		2023-02-27    AAPL     185.00   4
		2023-02-28    AAPL     185.00   4
		2022-12-28    AAPL     185.00   4
        2023-01-30    AAPL     185.00   5
    """
   
    return stock_data

def add_week_of_year(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
	A numeric value respresenting the week of the year (1 = first week, 2 = second week, ...)
    for each stock in the DataFrame.

    Parameters:
        stock_data (pd.DataFrame): A DataFrame containing stock data with the following columns:
            - 'Symbol': The stock symbol or identifier for each data record.
            - 'Date': The date for each data record. Should be set as the DataFrame index.
            - 'Adj Close': The adjusted closing price of the stock for each data record.

    Returns:
        pd.DataFrame: A new DataFrame with the 'WeekOfYear' field added.
            The returned DataFrame will be the same as the input 'stock_data'
            but with an additional column named 'WeekOfYear'.
            The 'WeekOfYear' column will contain the number respresenting the week of the year
            for the respective stock, week, and year combination. For non-trading days or 
			missing data, the 'WeekOfYear' value will be 0.

    Note:
        The 'Date' column should be set as the DataFrame index, and the data should be sorted in
        chronological order based on the date for correct results.

    Example:
        >>> stock_data = pd.DataFrame({
        ...     'Symbol':    ['AAPL',       'AAPL',       'AAPL',       'GOOGL',      'GOOGL',      'GOOGL',      'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL'],
        ...     'Date':      ['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-03', '2022-01-04', '2022-01-06', 2023-01-17, 2023-01-18, 2023-01-19, 2023-02-21, 2023-02-22, 2023-02-23, 2023-02-27, 2023-02-28],
        ...     'Adj Close': [180.05,       182.21,       None,         3100.0,       3095.5,       3134.0,       180.00,     181.00,     182.00,     183.00,     184.00,     185.00,     185.00,     185.00]
        ... })
        >>> stock_data.set_index('Date', inplace=True)
        >>> result = add_trading_days_in_year(stock_data)
        >>> print(result)
                     Symbol  Adj Close  WeekOfYear
        Date
        2022-01-03    AAPL     180.05   1
        2022-01-04    AAPL     182.21   1
        2022-01-05    AAPL       NaN    0
        2022-01-03   GOOGL    3100.00   1
        2022-01-04   GOOGL    3095.50   1
        2022-01-06   GOOGL    3134.00   1
        2023-01-17    AAPL     180.00   3
        2023-01-18    AAPL     181.00   3
        2023-01-19    AAPL     182.00   3
        2023-02-21    AAPL     183.00   8
        2023-02-22    AAPL     184.00   8
        2023-02-23    AAPL     185.00   8
		2023-02-27    AAPL     185.00   9
		2023-02-28    AAPL     185.00   9
    """

    # Get the ISO week number and year for each date
    iso_calendar = stock_data.index.isocalendar()
    stock_data['WeekOfYear'] = iso_calendar.week

    # Update the index name to 'Date' (if needed)
    # stock_data.index.name = 'Date'

    return stock_data

def add_month_of_year(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    A numeric value respresenting the month of the yearh (1 = first month, 2 = second month, ...) 
    for each stock in the DataFrame.

    Parameters:
        stock_data (pd.DataFrame): A DataFrame containing stock data with the following columns:
            - 'Symbol': The stock symbol or identifier for each data record.
            - 'Date': The date for each data record. Should be set as the DataFrame index.
            - 'Adj Close': The adjusted closing price of the stock for each data record.

    Returns:
        pd.DataFrame: A new DataFrame with the 'MonthOfYear' field added.
            The returned DataFrame will be the same as the input 'stock_data'
            but with an additional column named 'MonthOfYear'.
            The 'MonthOfYear' column will contain the number of the month

    Note:
        The 'Date' column should be set as the DataFrame index, and the data should be sorted in
        chronological order based on the date for correct results.

    Example:
        >>> stock_data = pd.DataFrame({
        ...     'Symbol':    ['AAPL',       'AAPL',       'AAPL',       'GOOGL',      'GOOGL',      'GOOGL',      'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL',     'AAPL'],
        ...     'Date':      ['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-03', '2022-01-04', '2022-01-06', 2023-01-17, 2023-01-18, 2023-01-19, 2023-02-21, 2023-02-22, 2023-02-23],
        ...     'Adj Close': [180.05,       182.21,       None,         3100.0,       3095.5,       3134.0,       180.00,     181.00,     182.00,     183.00,     184.00,     185.00]
        ... })
        >>> stock_data.set_index('Date', inplace=True)
        >>> result = add_trading_days_in_year(stock_data)
        >>> print(result)
                     Symbol  Adj Close  MonthOfYear
        Date
        2022-01-03    AAPL     180.05   1
        2022-01-04    AAPL     182.21   1
        2022-01-05    AAPL       NaN    0
        2022-01-03   GOOGL    3100.00   1
        2022-01-04   GOOGL    3095.50   1
        2022-01-06   GOOGL    3134.00   1
        2023-01-17    AAPL     180.00   1
        2023-01-18    AAPL     181.00   1
        2023-01-19    AAPL     182.00   1
        2023-02-21    AAPL     183.00   2
        2023-02-22    AAPL     184.00   2
        2023-02-23    AAPL     185.00   2
    """
	
    stock_data['MonthOfYear'] = stock_data.index.month
    return stock_data
