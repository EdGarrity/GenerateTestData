"""
Write a Python function which does the following:
1) Read a csv file which contains GDP data.  The first column in the file is 
   the date.  The second column in the file is the GDP value.
2) The function receives a reference to the test_data panda data frame table.  
   The test_data table has four columns.  The first column is the name of the 
   stock. The second column is the date. The thrird column is the key.  The 
   fourth column is the value.
3) The function will add a row in the test_data table for every business day in 
   the range provided to the function.  For each row created, the function will 
   store the string "GDP" in the first and third column.  The function will 
   store the most recent GDP value with respect to the date of the row in the 
   third column.  If the GDP table does not have a value for that date, the 
   function will use the previous value in the GDP table.
4) The function will return the updated test_data table.
"""
import pandas as pd


def add_gdp_data(test_data: pd.DataFrame, gdp_data_filename: str):
    """
    Adds GDP data to the test_data Pandas DataFrame.

    Args:
        test_data: The Pandas DataFrame to add the GDP data to.
        gdp_data_filename: The CSV file containing the GDP data.
        
    This function reads the GDP data into a Pandas DataFrame, gets the start 
    and end dates of the test data, and then iterates over the dates in the 
    range of the test data. For each date, the function checks if it is a 
    business day. If it is, the function gets the most recent GDP value for the 
    date and adds a row to the test_data DataFrame with the stock "GDP", the 
    date, the key "GDP", and the GDP value. If the GDP table does not have a 
    value for the date, the function uses the previous value in the GDP table.
    """

    # Read the GDP data into a Pandas DataFrame.
    gdp_df = pd.read_csv(gdp_data_filename)

    # Get the start and end dates of the test data.
    start_date = test_data['date'].min()
    end_date = test_data['date'].max()

    # Iterate over the dates in the range of the test data.
    for date in pd.date_range(start_date, end_date):
        # Get the most recent GDP value for the date.
        most_recent_gdp_value = gdp_df[gdp_df['date'] <= date].sort_values(
            'date', ascending=False)['gdp'].values[0]

        # Add a row to the test_data DataFrame for the date.
        row = {
            'stock': 'GDP',
            'date': date,
            'key': 'GDP',
            'value': most_recent_gdp_value
        }
        test_data = test_data.append(row, ignore_index=True)

    return test_data
