import pandas as pd

def add_temporal_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds temporal data to the input DataFrame.

    The input stock_data DataFrame consists of stock information for several stocks from the first trading day of 1947 to the last trading day of 2022.  The DataFrame comtains many columns.  The name of the stock is in the 'Stock' column of the DataFrame.  The date is the row.name of the DataFrame

    For each record in the DataFrame, this function will add the following fields:

    TradingDaysInYear
        Calculate TradingDaysInYear by counting how many days the DataFrame has information for that stock in the given year from the first record to the last record for this stock for the year.

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
    """

    print("DataFrame structure:")
    print(stock_data.info())
    print()
    print("DataFrame columns:")
    for col in stock_data.columns:
        print(f"  {col}: {stock_data[col].dtype}")
        print(f"    index: {stock_data.index.dtype}")

    stock_data_copy = stock_data.copy()
    
    # Save Date index
    date_index = stock_data_copy.index

    # Reset index
    stock_data_copy = stock_data_copy.reset_index(drop=True)

    # Re-add Date column
    stock_data_copy['Date'] = date_index

    # Convert Date to datetime
    stock_data_copy['Date'] = pd.to_datetime(stock_data_copy['Date'])

    # Calculate trading days
    days = (stock_data_copy.groupby([stock_data_copy['Date'].dt.year, 'Stock'])
            .size()
            .reset_index(name='TradingDaysInYear'))

    # Merge on Stock
    stock_data = stock_data.merge(days, on=['Stock'], how='left')

    return stock_data


def add_trading_days_in_year(stock_data: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds the `TradingDaysInYear` field to the DataFrame.
    '''
    
    # Add the `index` column to the DataFrame.
    stock_data['index'] = stock_data.index

    # Get the year for each record in the DataFrame.
    stock_data['year'] = stock_data['index'].dt.year

    # Initialize a dictionary to store the number of trading days for each stock and year.
    trading_days_in_year = {}

    # Iterate over the DataFrame and update the dictionary with the number of trading days for each stock and year.
    for stock, group in stock_data.groupby('Stock'):
        for year, df_year in group.groupby('year'):
            trading_days_in_year[(stock, year)] = len(df_year)

    # Add the `TradingDaysInYear` field to the DataFrame.
    stock_data['TradingDaysInYear'] = stock_data[['Stock', 'year']].apply(
        lambda x: trading_days_in_year[x[0], x[1]], axis=1)

    return stock_data
