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
    test_data = src.kv_collection.load_stock_data(stock_data, test_data)
    test_data = src.add_gdp_data.add_gdp_data(test_data, 'gdp_data.csv')
    test_data = src.add_cpi_data.add_cpi_data(test_data, 'CPILFENS.csv')
    src.kv_collection.save_to_sql(test_data)

    print(stock_data)
    
# import yfinance as yf
# import pandas as pd

# def load_index_data(index_symbols, start_date, end_date):
#     """
#     This function downloads stock data and normalizes the data.
#     :param index_symbols: list of index symbols
#     :param start_date: start date
#     :param end_date: end date
#     :return: dataframe
#     """
#     # download data
#     data = pd.DataFrame()
#     for stock in index_symbols:
#         ticker_data = yf.download(stock, start=start_date, end=end_date)
#         ticker_data['Symbol'] = stock
#         data = pd.concat([data, ticker_data], axis=0)

#     # Set the DataFrame index to be the row names (date)
#     data.index.name = 'Date'

#     # adjust all OHLC using the row names (date)
#     data['Adj_Open'] = data['Open'] / data['Close'] * data['Adj Close']
#     data['Adj_High'] = data['High'] / data['Close'] * data['Adj Close']
#     data['Adj_Low'] = data['Low'] / data['Close'] * data['Adj Close']
#     data['Adj_Volume'] = data['Volume'] / data['Adj Close'] * data['Close']

#     for stock in index_symbols:
#         mask = data['Symbol'] == stock

#         # The normalization calculations now use the row names (date) instead of 'Date' column
#         data.loc[mask, 'Norm_Adj_Close']  \
#             = (data.loc[mask, 'Adj Close'] - data.loc[mask, 'Adj_Low'].min()) \
#             / (data.loc[mask, 'Adj_High'].max() - data.loc[mask, 'Adj_Low'].min())

#         data.loc[mask, 'Norm_Adj_High']  \
#             = (data.loc[mask, 'Adj_High'] - data.loc[mask, 'Adj_Low'].min()) \
#             / (data.loc[mask, 'Adj_High'].max() - data.loc[mask, 'Adj_Low'].min())

#         data.loc[mask, 'Norm_Adj_Low']  \
#             = (data.loc[mask, 'Adj_Low'] - data.loc[mask, 'Adj_Low'].min()) \
#             / (data.loc[mask, 'Adj_High'].max() - data.loc[mask, 'Adj_Low'].min())

#         data.loc[mask, 'Norm_Adj_Open']  \
#             = (data.loc[mask, 'Adj_Open'] - data.loc[mask, 'Adj_Low'].min()) \
#             / (data.loc[mask, 'Adj_High'].max() - data.loc[mask, 'Adj_Low'].min())

#         data.loc[mask, 'Norm_Adj_Volume']  \
#             = (data.loc[mask, 'Adj_Volume'] - data.loc[mask, 'Adj_Volume'].min()) \
#             / (data.loc[mask, 'Adj_Volume'].max() - data.loc[mask, 'Adj_Volume'].min())

#     return data

# def add_trading_days_in_year(stock_data: pd.DataFrame) -> pd.DataFrame:
#     """
#     This function calculates the `TradingDaysInYear` for each stock in the DataFrame.

#     Parameters:
#         stock_data (pd.DataFrame): A DataFrame with stock data.

#     Returns:
#         pd.DataFrame: The DataFrame with the `TradingDaysInYear` field added.
#     """
#     # Group data by stock symbol and year
#     grouped_data = stock_data.groupby([stock_data.index.year, 'Symbol'])

#     # Calculate TradingDaysInYear for each stock and year
#     stock_data['TradingDaysInYear'] = grouped_data['Adj Close'].transform(lambda x: x.notnull().sum())

#     return stock_data

# def add_stock_temporal_data() -> pd.DataFrame:
#     """
#     This program downloads stock data and adds the `TradingDaysInYear` field.

#     For each record in the downloaded stock data, add the following field:
#     - TradingDaysInYear: Calculate the number of trading days in the year for each stock.

#     Parameters:
#         None

#     Returns:
#         pd.DataFrame: A DataFrame with the stock data including the added temporal-related information.
#     """
#     # Download data
#     stock_data = load_index_data(['AAPL', 'FXAIX'], '2019-01-01', '2021-12-31')

#     # Add TradingDaysInYear field
#     stock_data = add_trading_days_in_year(stock_data)

#     # Return stock data with additional temporal-related information
#     return stock_data

# if __name__ == '__main__':
#     stock_data_2 = add_stock_temporal_data()
#     print(stock_data_2)
