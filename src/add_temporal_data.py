'''
    Adds temporal data to the input DataFrame.

    The input test_data DataFrame consists of stock information from the first trading day of 1947 to the last trading day of 2022 in the following format:
        'Stock': The stock symbol
        'Date': The date
        'Key': The string representing the dimension
        'Value': A double that represents the value of the dimension

    For each stock and date in the 'test_data' DataFrame, this function will create a new record with the following structure:
        'Stock': The stock symbol
        'Date': The date
        'Key': The string 'TradingDaysSinceStartOfYear'
        'Value': TradingDaysSinceStartOfYear
            Where TradingDaysSinceStartOfYear is calculated by counting how many days the DataFrame has information for that stock in the given year, not how many calendar days are in the year.

    For each stock and date in the 'test_data' DataFrame, this function will create a new record with the following structure:
        'Stock': The stock symbol
        'Date': The date
        'Key': The string 'TradingDaysLeftInYear'
        'Value': TradingDaysLeftInYear
            Where TradingDaysLeftInYear is calculated by counting how many days the DataFrame has information for that stock in the given year from the current record to the last record for this stock for the year.
            
            The TradingDaysLeftInYear is calculated as follows:
            1) Calculate TradingDaysInYear by counting how many days the DataFrame has information for that stock in the given year from the first record to the last record for this stock for the year.
            2) Calculate TradingDaysLeftInYear as TradingDaysInYear - TradingDaysSinceStartOfYear
        
    For each stock and date in the 'test_data' DataFrame, this function will create a new record with the following structure:
        'Stock': The stock symbol
        'Date': The date
        'Key': The string 'TradingDaysInWeek'
        'Value': TradingDaysInWeek
            Where TradingDaysInWeek is calculated by counting how many days the DataFrame has information for that stock in the given week in the year.
        

    Parameters:
        test_data (pd.DataFrame): The DataFrame containing stock market or financial data.  The structure of the DataFrame is as follows:
            'Stock': A string that represents the stock trading symbol
            'Date': Column representing dates in a valid date format, compatible with the 'pd.to_datetime()' function
            'Key': A string that represents the dimension of the data
            'Value': A double that represents the value of the dimension.
    
    Returns:
        pd.DataFrame: A DataFrame with the updated 'test_data' including the added temporal-related information.

    Example:
        Suppose we have a DataFrame 'stock_data' with columns 'Date' and 'Stock_Price',
        and we want to add temporal data to it.  We can use the function as follows:

        >>> updated_data = add_temporal_data(stock_data)
        >>> print(updated_data)

    Note:
        - The 'Date' column in the 'test_data' DataFrame should be in a valid date format,
          compatible with the 'pd.to_datetime()' function used in this function.
        - The 'test_data' DataFrame will be modified in place to include the additional temporal data.
'''
import pandas as pd
from datetime import datetime

def add_temporal_data(test_data: pd.DataFrame) -> pd.DataFrame:

    # Convert Date to datetime
    test_data['Date'] = pd.to_datetime(test_data['Date'])

    # Initialize dictionaries to store counts 
    trading_days_in_year = {}
    trading_days_since_start_of_year = {}
    trading_days_in_week = {}

    # Calculate the number of trading days for each year and trading days in each week
    for index, row in test_data.iterrows():

        # Extract info
        stock = row['Stock']
        date = row['Date']
        year = date.year
        week = date.weekofyear

        # Calculate total trading days in year count
        if stock not in trading_days_in_year:
            trading_days_in_year[stock] = {year: 0}
        if row['Key'] == 'Close':
            trading_days_in_year[stock][year] += 1

        # Update trading days in week count
        if stock not in trading_days_in_week:
            trading_days_in_week[stock] = {}
        if year not in trading_days_in_week[stock]:
            trading_days_in_week[stock][year] = {}
        if row['Key'] == 'Close':
            trading_days_in_week[stock][year][week] += 1

    # Iterate through each row
    for index, row in test_data.iterrows():

        # Extract info
        stock = row['Stock']
        date = row['Date']
        year = date.year
        week = date.weekofyear

        # Update trading days count
        if stock not in trading_days_in_year:
            trading_days_since_start_of_year[stock] = {year: 0}
        if row['Key'] == 'Close':
            trading_days_since_start_of_year[stock][year] += 1

        # Create new row for TradingDaysSinceStartOfYear
        new_row = {
        'Stock': stock,
        'Date': date,
        'Key': 'TradingDaysSinceStartOfYear',
        'Value': trading_days_since_start_of_year[stock][year]
        }

        # Append new row for TradingDaysSinceStartOfYear
        test_data = test_data.append(new_row, ignore_index=True)

        # Create new row for TradingDaysLeftInYear
        new_row = {
        'Stock': stock,
        'Date': date,
        'Key': 'TradingDaysSinceStartOfYear',
        'Value': trading_days_in_year[stock][year] - trading_days_since_start_of_year[stock][year]
        }

        # Append new row for TradingDaysLeftInYear
        test_data = test_data.append(new_row, ignore_index=True)

        # Calculate the number of trading days in the week for the stock
    
        # Create new row for TradingDaysInWeek
        new_row = {
        'Stock': stock,
        'Date': date,
        'Key': 'TradingDaysInWeek',
        'Value': trading_days_in_week[stock][year][week]
        }

        # Append new row for TradingDaysInWeek
        test_data = test_data.append(new_row, ignore_index=True)

    return test_data
  