import pandas as pd
import numpy as np

# data = pd.read_csv('stock_data_2018.csv')
# tickers = data['Ticker'].unique()

data1 = pd.read_csv('data_log_return.csv')
tickers1 = list(data1.columns)

data2 = pd.read_csv('dates_log.csv')
tickers2 = list(data2.columns)

if tickers1 == tickers2:
    print('Тикеры совпадают.')
else:
    print('Тикеры не совпадают.')

# # Создаем пустой DataFrame, в котором даты будут индексами
# new_data = pd.DataFrame()
# new_data['Date'] = sorted(list(data['Date'].unique()))
# new_data = new_data.set_index('Date')

# '''____________________________________________________________________________________________________________

# Проходимся по каждому тикеру и создаем табличку с данными закрытия торгов для этого тикера.
# Из столбца 'Date' снова делаем индексы.
# Считаем логарифмическую доходность по каждому дню, используя shift.
# Теперь, когда мы добавляем получившиеся значения в new_data, данные раскидываются по соответствующим датам.
# Если в какой-то день торгов не было, то ячейка остается незаполненной.
# ____________________________________________________________________________________________________________'''

# for ticker in tickers:
#     current_ticker = pd.DataFrame()
#     current_ticker['Date'] = data[data['Ticker'] == ticker]['Date']
#     if current_ticker.shape[0] < 200:
#         continue
#     current_ticker['Close'] = data[data['Ticker'] == ticker]['Close']
#     current_ticker = current_ticker.set_index('Date')
    
#     new_data[ticker] = np.log(current_ticker['Close'] / current_ticker['Close'].shift(1))

# print(new_data)

#new_data.to_csv('data_log_return.csv')
