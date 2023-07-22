"""
Write a Python function which does the following: 
1) Read a csv text file which contains GDP data. The first column in the file 
is a string named 'date'. The second column in the file is a double named 
'value'. The first column needs to be converted from a string to a date.  Here 
are some samples of the dates in the first column: '7/1/2021', '10/1/2021', 
'10/1/2022'.
2) The function receives a reference to the test_data panda data frame table. 
The test_data table has four columns. The first column is a string named 
'Stock'. The second column is of type date and is named 'date'. The third 
column is a string named 'key'. The fourth column is a double named 'value'. 
3) For each row in the provided test_data table the function will add a row in 
the new_test_data table with the first column of the new row equal to the 
string 'GDP', the second columnd of the new row equal to the date from the 
test_date date column, the third column of the new row will be equal to the 
string 'GDP', the fourth column of the new row will be the equal to the value 
of the GDP with the same dat as the test_data row. If the GDP table does not 
have a value for that date, the function will use the previous value in the GDP 
table. 
4) The function will return the new_test_data table.
"""
import pandas as pd

def add_gdp_data(test_data: pd.DataFrame, gdp_data_filename: str) -> pd.DataFrame:
    """
    Adds GDP-related data to the input DataFrame.

    This function reads GDP data from a CSV file specified by 'gdp_data_filename',
    and for each date in the 'test_data' DataFrame, it retrieves the most recent GDP value
    with respect to that date and calculates the year-over-year percent change in GDP.
    The calculated GDP information is then added to the 'test_data' DataFrame.

    Parameters:
        test_data (pd.DataFrame): The DataFrame containing stock market or financial data,
                                  with a 'Date' column representing dates.
        gdp_data_filename (str): The filename of the CSV file containing GDP data.
    
    Returns:
        pd.DataFrame: A DataFrame with the updated 'test_data' including added GDP-related information.

    Example:
        Suppose we have a DataFrame 'stock_data' with columns 'Date' and 'Stock_Price',
        and we want to add GDP data to it from the file 'gdp_data.csv'.
        We can use the function as follows:

        >>> updated_data = add_gdp_data(stock_data, 'gdp_data.csv')
        >>> print(updated_data)

    Note:
        - The CSV file specified by 'gdp_data_filename' should contain at least two columns:
          'date' and 'value', representing GDP data and corresponding dates respectively.
        - The 'Date' column in the 'test_data' DataFrame should be in a valid date format,
          compatible with the 'pd.to_datetime()' function used in this function.
        - The 'test_data' DataFrame will be modified in place to include the additional GDP data.
    """

    # Read GDP data from the CSV file
    gdp_data = pd.read_csv(gdp_data_filename)

    # Convert the 'date' column to datetime
    gdp_data['date'] = pd.to_datetime(gdp_data['date'])

    # Sort the GDP data by date in ascending order
    gdp_data = gdp_data.sort_values('date')

    # Retrieve list of dates from test_data
    dates = test_data['Date'].unique()
    
    # Iterate over each date in the test_data DataFrame
    for date in dates:
        # Find the most recent GDP value with respect to the date
        recent_gdp = gdp_data[gdp_data['date'] <= date]['value'].iloc[-1]

        # Create a new row for GDP data for the current value
        new_row = pd.DataFrame({
            'Stock': 'GDP',
            'Date': date,
            'Key': 'Close',
            'Value': recent_gdp
        }, index=[0])

        # Concatenate the new row to new_test_data DataFrame
        test_data = pd.concat([test_data, new_row], ignore_index=True)

        # Find the GDP value from the previous year
        previous_year_gdp = gdp_data[gdp_data['date'] <= date - pd.DateOffset(years=1)]['value'].iloc[-1]

        # Calculate the year-over-year percent change
        percent_change = (recent_gdp - previous_year_gdp) / \
            previous_year_gdp * 100

        # Create a new row for GDP data with percent change
        new_row = pd.DataFrame({
            'Stock': 'GDP',
            'Date': date,
            'Key': 'Yearly Percent Change',
            'Value': percent_change
        }, index=[0])

        # Concatenate the new row to new_test_data DataFrame
        test_data = pd.concat([test_data, new_row], ignore_index=True)

    return test_data
