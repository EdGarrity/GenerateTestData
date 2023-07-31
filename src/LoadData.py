import pandas as pd

def LoadData(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Print the contents of the DataFrame as a check
    print(df)

    # Return the DataFrame
    return df

data_frame = LoadData("InterationOfDayOfWeek.csv")
