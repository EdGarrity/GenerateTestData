"""
This program downloads stock data, normalizes the data, and saves it to a MS SQL database
"""
import pandas as pd
import src.stock_data
import src.technical_analysis
import src.kv_collection



if __name__ == '__main__':
    test_data = pd.DataFrame(columns=['Stock', 'Date', 'Key', 'Value'])

    stock_data = src.stock_data.get()
    stock_data = src.technical_analysis.generate(stock_data)
    test_data = src.kv_collection.load_stock_data(stock_data, test_data)
    src.kv_collection.save_to_sql(test_data)

    print (stock_data.columns.tolist())