"""
This program downloads stock data, normalizes the data, and saves it to a MS SQL database
"""
import pandas as pd
import src.stock_data
import src.technical_analysis
import src.kv_collection
import src.add_gdp_data
import src.add_cpi_data
import src.temporal_data


if __name__ == '__main__':
    test_data = pd.DataFrame(columns=['Stock', 'Date', 'Key', 'Value'])

    stock_data = src.stock_data.get()
    stock_data = src.technical_analysis.generate(stock_data)
    stock_data = src.temporal_data.add_trading_days_in_year(stock_data)
    stock_data = src.temporal_data.add_trading_days_since_start_of_year(stock_data)
    test_data = src.kv_collection.load_stock_data(stock_data, test_data)
    test_data = src.add_gdp_data.add_gdp_data(test_data, 'gdp_data.csv')
    test_data = src.add_cpi_data.add_cpi_data(test_data, 'CPILFENS.csv')
    src.kv_collection.save_to_sql(test_data)

    print(stock_data)
    

# def add_trading_days_since_start_of_year(stock_data: pd.DataFrame) -> pd.DataFrame:
#     # Step 1: Extract the year from the dates
#     stock_data['Year'] = pd.to_datetime(stock_data.index).year

#     # Step 2: Group by 'Symbol' and 'Year' and calculate the number of trading days since the start of the year
#     stock_data['TradingDaysSinceStartOfYear'] = stock_data.groupby(['Symbol', 'Year'])['Adj Close'].apply(lambda x: x.notnull().cumsum())

#     # Drop the 'Year' column as it is no longer needed
#     stock_data.drop(columns=['Year'], inplace=True)

#     return stock_data

# # Test Example
# stock_data = pd.DataFrame({
#     'Symbol':    ['AAPL',       'AAPL',       'AAPL',       'AAPL',       'AAPL',       'GOOGL',      'GOOGL',      'GOOGL',      'GOOGL',      'GOOGL'],
#     'Date':      ['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-03', '2022-01-04', '2022-01-05', '2023-01-02', '2023-01-03'],
#     'Adj Close': [ 180.05,       182.21,       None,         182.22,       182.21,       3100.0,       3095.5,       3134.0,       3144.00,      3144.00]
# })
# stock_data.set_index('Date', inplace=True)
# result = add_trading_days_since_start_of_year(stock_data)
# print(result)
