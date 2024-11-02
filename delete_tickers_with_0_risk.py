import pandas as pd
import numpy as np

pd.set_option('display.max_rows',None)

data = pd.read_csv('data_log_return.csv')
tickers = list(data.columns)
tickers.pop(0)

data['Date'] = sorted(list(data['Date'].unique()))
data = data.set_index('Date')

# Эти тикеры мы взяли из функции find_E_n_sigma в файле code_part, дописав условие поиска тикеров с нулевым риском.
to_delete = ['AHEB5.SA', 'AHEB6.SA', 'BDLL3.SA', 'CASN3.SA', 'FAED11.SA',
             'FDES11.SA', 'FIGE3.SA', 'HOOT3.SA', 'KINP11.SA', 'LATR11B.SA',
             'MAPT3.SA', 'ODER4.SA', 'PHMO34.SA', 'PINE3.SA', 'REIT11.SA',
             'SHUL3.SA', 'SNSY3.SA', 'SNSY6.SA', 'SOND3.SA']

data = data.drop(to_delete, axis=1)

data.to_csv('data_log_return.csv')