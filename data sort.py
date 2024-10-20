import pandas as pd
import numpy as np

data = pd.read_csv('stock_data_2018.csv')
tickers = data['Ticker'].unique() # Create a list of unique tickers names

# Create a DataFrame to store the closing prices for each ticker
dates = pd.DataFrame()
dates['Date'] = sorted(list(data['Date'].unique()))
dates = dates.set_index('Date')

dates_log = dates

'''__________________________________________________________________________________

Проходимся по каждому тикеру и 
__________________________________________________________________________________'''

for ticker in tickers[:1]:
    current_ticker = pd.DataFrame()
    current_ticker = current_ticker.assign(data[data['Ticker'] == ticker]['Date'], data[data['Ticker'] == ticker]['Close'])
    # current_ticker['Date'] = data[data['Ticker'] == ticker]['Date']
    # current_ticker['Close'] = data[data['Ticker'] == ticker]['Close']
    print(current_ticker)
    # current_ticker = current_ticker.set_index('Date')

    # dates[ticker] = current_ticker['Close']

    # test = pd.DataFrame()
    # test['log_return'] = np.log(dates[ticker] / dates[ticker].shift(1))
    # dates_log[ticker] = test['log_return']


# data.to_csv('test.csv')