import pandas as pd
import numpy as np

data = pd.read_csv('stock_data_2018.csv')
tickers = data['Ticker'].unique()

# Создаем пустой DataFrame, в котором даты будут индексами
new_data = pd.DataFrame()
new_data['Date'] = sorted(list(data['Date'].unique()))
new_data = new_data.set_index('Date')

'''____________________________________________________________________________________________________________

Проходимся по каждому тикеру и создаем табличку с данными закрытия каждого дня торгов для этого актива.
Из столбца 'Date' снова делаем индексы.
Считаем логарифмическую доходность по каждому дню, используя shift.
Теперь, когда мы добавляем получившиеся значения в new_data, данные раскидываются по соответствующим датам.
Если в какой-то день торгов не было, то ячейка остается незаполненной (NaN).
____________________________________________________________________________________________________________'''

for ticker in tickers:
    current_ticker = pd.DataFrame()
    current_ticker['Date'] = data[data['Ticker'] == ticker]['Date']
    if current_ticker.shape[0] < 200: # Если у актива меньше 200 дней торгов, то не добавляем его в табличку
        continue
    current_ticker['Close'] = data[data['Ticker'] == ticker]['Close']
    current_ticker = current_ticker.set_index('Date')
    
    new_data[ticker] = np.log(current_ticker['Close'] / current_ticker['Close'].shift(1))
    
new_data.to_csv('data_log_return.csv')
