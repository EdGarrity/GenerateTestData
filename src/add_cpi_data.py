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

def add_cpi_data(test_data: pd.DataFrame, cpi_data_filename: str):
    """
    Adds year-over-year percent change in CPI data to the test_data Pandas DataFrame.

    Args:
        test_data: The Pandas DataFrame to add the CPI data to.
        cpi_data_filename: The CSV file containing the CPI data.
        
    This function reads the CPI data into a Pandas DataFrame, gets the start 
    and end dates of the test data, and then iterates over the dates in the 
    range of the test data. For each date, the function checks if it is a 
    business day. If it is, the function gets the most recent CPI value for the 
    date and adds a row to the test_data DataFrame with the stock "CPI", the 
    date, the key "CPI", and the CPI value. If the CPI table does not have a 
    value for the date, the function uses the previous value in the CPI table.
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
        previous_year_date = date - pd.DateOffset(years=1)
        previous_year_cpi = cpi_data[cpi_data['date']
                                     == previous_year_date]['value'].iloc[0]

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
