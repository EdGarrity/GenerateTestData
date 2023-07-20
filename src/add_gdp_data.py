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

def add_gdp_data(test_data: pd.DataFrame, gdp_data_filename: str):
    """
    Adds year-over-year percent change in GDP data to the test_data Pandas DataFrame.

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

    # Read GDP data from the CSV file
    gdp_data = pd.read_csv(gdp_data_filename)

    # Convert the 'date' column to datetime
    gdp_data['date'] = pd.to_datetime(gdp_data['date'])

    # Sort the GDP data by date in ascending order
    gdp_data = gdp_data.sort_values('date')

    # Create a new DataFrame for updated test data
    # new_test_data = pd.DataFrame(columns=['Stock', 'date', 'key', 'value'])

    # Retrieve list of dates from test_data
    dates = test_data['Date'].unique()
    
    # Iterate over each date in the test_data DataFrame
    for date in dates:
        # Find the most recent GDP value with respect to the date
        recent_gdp = gdp_data[gdp_data['date'] <= date]['value'].iloc[-1]

        # Create a new row for GDP data with percent change
        new_row = pd.DataFrame({
            'Stock': 'GDP',
            'Date': date,
            'Key': 'Close',
            'Value': recent_gdp
        }, index=[0])

        # Concatenate the new row to new_test_data DataFrame
        test_data = pd.concat([test_data, new_row], ignore_index=True)

        # Find the GDP value from the previous year
        previous_year_date = date - pd.DateOffset(years=1)
        previous_year_gdp = gdp_data[gdp_data['date']
                                     == previous_year_date]['value'].iloc[0]

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
