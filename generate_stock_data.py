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
    # stock_data = src.technical_analysis.generate(stock_data)
    stock_data = src.temporal_data.add_trading_days_in_year(stock_data)
    stock_data = src.temporal_data.add_trading_days_since_start_of_year(stock_data)
    stock_data = src.temporal_data.add_trading_days_left_in_year(stock_data)
    stock_data = src.temporal_data.add_trading_days_in_week(stock_data)
    stock_data = src.temporal_data.add_day_of_week(stock_data)
    stock_data = src.temporal_data.add_month_of_year(stock_data)
    stock_data = src.temporal_data.add_week_of_year(stock_data)
    test_data = src.kv_collection.load_stock_data(stock_data, test_data)
    test_data = src.add_gdp_data.add_gdp_data(test_data, 'gdp_data.csv')
    test_data = src.add_cpi_data.add_cpi_data(test_data, 'CPILFENS.csv')
    # src.kv_collection.save_to_sql(test_data)

    print(stock_data)
