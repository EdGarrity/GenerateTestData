import pandas as pd

def add_temporal_data(test_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds temporal data to the input DataFrame.

    For each stock and date in the 'test_data' DataFrame, this function will add the following records to the 'test_data' DataFrame:
    
    1) A record representing the day of the week:
		'Stock': The stock symbol
		'Date': The date
		'Key': The string 'DayOfWeek'
		'Value': A double that represents the day of the week (1 = Sunday, 2 = Monday, ...)
	
    2) A record representing the day of the week:
		'Stock': The stock symbol
		'Date': The date
		'Key': The string 'DayOfWeek'
		'Value': A double that represents the day of the week (1 = Sunday, 2 = Monday, ...)
	
	
	calculate the day of the week (1 = Sunday, 2 = Monday, ...), the number of trading days since the start of the year, the number of trading days left in the year, and the month of the year (1 = January, 2 = February, ...).  The calculated temporal information is then added to the 'test_data' DataFrame.

    Parameters:
        test_data (pd.DataFrame): The DataFrame containing stock market or financial data.  The structure of the DataFrame is as follows:
			'Stock': A string that represents the stock trading symbol
            'Date': Column representing dates in a valid date format, compatible with the 'pd.to_datetime()' function
            'Key': A string that represents the dimension of the data
            'Value': A double that represents the value of the dimension.
    
    Returns:
        pd.DataFrame: A DataFrame with the updated 'test_data' including added temporal-related information.

    Example:
        Suppose we have a DataFrame 'stock_data' with columns 'Date' and 'Stock_Price',
        and we want to add temporal data to it.  We can use the function as follows:

        >>> updated_data = add_temporal_data(stock_data)
        >>> print(updated_data)

    Note:
        - The 'Date' column in the 'test_data' DataFrame should be in a valid date format,
          compatible with the 'pd.to_datetime()' function used in this function.
        - The 'test_data' DataFrame will be modified in place to include the additional temporal data.
    """

    # Convert the 'Date' column to a pandas datetime object
    test_data['Date'] = pd.to_datetime(test_data['Date'])

    # Calculate day of the week (1 = Sunday, 2 = Monday, ...)
    test_data['DayOfWeek'] = test_data['Date'].dt.dayofweek + 1

    # Calculate the number of trading days since the start of the year
    test_data['TradingDaysSinceStartOfYear'] = (test_data['Date'] -
                                                test_data['Date'].dt.to_period('Y').dt.start_time).dt.days

    # Calculate the number of trading days left in the year
    test_data['TradingDaysLeftInYear'] = (test_data['Date'].dt.to_period('Y').dt.end_time - test_data['Date']).dt.days

    # Calculate the month of the year (1 = January, 2 = February, ...)
    test_data['MonthOfYear'] = test_data['Date'].dt.month

    return test_data

# Example usage:
# Suppose you have a DataFrame 'stock_data' with columns 'Date' and 'Stock_Price',
# and you want to add temporal data to it.
# updated_data = add_temporal_data(stock_data)
# print(updated_data)
