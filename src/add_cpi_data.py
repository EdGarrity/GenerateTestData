"""
Write a Python function which does the following: 
1) Read a csv text file which contains CPI data. The first column in the file 
is a string named 'date'. The second column in the file is a double named 
'value'. The first column needs to be converted from a string to a date.  Here 
are some samples of the dates in the first column: '1967-12-01', '2023-06-01'.
2) The function receives a reference to the test_data panda data frame table. 
The test_data table has four columns. The first column is a string named 
'Stock'. The second column is of type date and is named 'date'. The third 
column is a string named 'key'. The fourth column is a double named 'value'. 
3) For each row in the provided test_data table the function will add a row in 
the new_test_data table with the first column of the new row equal to the 
string 'CPI', the second columnd of the new row equal to the date from the 
test_date date column, the third column of the new row will be equal to the 
string 'CPI', the fourth column of the new row will be the equal to the value 
of the CPI with the same dat as the test_data row. If the CPI table does not 
have a value for that date, the function will use the previous value in the CPI 
table. 
4) The function will return the new_test_data table.
"""
import pandas as pd

def add_cpi_data(test_data: pd.DataFrame, cpi_data_filename: str) -> pd.DataFrame:
    """
    Adds Consumer Price Index (CPI) related data to the input DataFrame.

    This function reads CPI data from a CSV file specified by 'cpi_data_filename',
    and for each date in the 'test_data' DataFrame, it retrieves the most recent CPI value
    with respect to that date and calculates the year-over-year percent change in CPI.
    The calculated CPI information is then added to the 'test_data' DataFrame.

    Parameters:
        test_data (pd.DataFrame): The DataFrame containing stock market or financial data,
                                  with a 'Date' column representing dates.
        cpi_data_filename (str): The filename of the CSV file containing CPI data.
    
    Returns:
        pd.DataFrame: A DataFrame with the updated 'test_data' including added CPI-related information.

    Example:
        Suppose we have a DataFrame 'stock_data' with columns 'Date' and 'Stock_Price',
        and we want to add CPI data to it from the file 'cpi_data.csv'.
        We can use the function as follows:

        >>> updated_data = add_cpi_data(stock_data, 'cpi_data.csv')
        >>> print(updated_data)

    Note:
        - The CSV file specified by 'cpi_data_filename' should contain at least two columns:
          'date' and 'value', representing CPI data and corresponding dates respectively.
        - The 'Date' column in the 'test_data' DataFrame should be in a valid date format,
          compatible with the 'pd.to_datetime()' function used in this function.
        - The 'test_data' DataFrame will be modified in place to include the additional CPI data.
    """
    # Read CPI data from the CSV file
    cpi_data = pd.read_csv(cpi_data_filename)

    # Convert the 'date' column to datetime
    cpi_data['date'] = pd.to_datetime(cpi_data['date'])

    # Sort the CPI data by date in ascending order
    cpi_data = cpi_data.sort_values('date')

    # Retrieve list of dates from test_data
    dates = test_data['Date'].unique()
    
    # Iterate over each date in the test_data DataFrame
    for date in dates:
        # Find the most recent CPI value with respect to the date
        recent_cpi = cpi_data[cpi_data['date'] <= date]['value'].iloc[-1]

        # Create a new row for CPI data for the current value
        new_row = pd.DataFrame({
            'Stock': 'CPI',
            'Date': date,
            'Key': 'Close',
            'Value': recent_cpi
        }, index=[0])

        # Concatenate the new row to new_test_data DataFrame
        test_data = pd.concat([test_data, new_row], ignore_index=True)

        # Find the GDP value from the previous year
        previous_year_cpi = cpi_data[cpi_data['date'] <= date - pd.DateOffset(years=1)]['value'].iloc[-1]

        # Calculate the year-over-year percent change
        percent_change = (recent_cpi - previous_year_cpi) / \
            previous_year_cpi * 100

        # Create a new row for CPI data with percent change
        new_row = pd.DataFrame({
            'Stock': 'CPI',
            'Date': date,
            'Key': 'Yearly Percent Change',
            'Value': percent_change
        }, index=[0])
       
        # Concatenate the new row to new_test_data DataFrame
        test_data = pd.concat([test_data, new_row], ignore_index=True)

    return test_data
