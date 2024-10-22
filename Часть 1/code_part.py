import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize

#pd.set_option('display.max_rows',None)

data = pd.read_csv("..\data_log_return.csv")
tickers = list(data.columns)
tickers.pop(0)

''' =========  № 1  =========================================================================================================== '''
'''
###      Оценка ожидаемых доходностей и стандартных отклонений.
'''
''' =========================================================================================================================== '''

# Вычисление оценки ожидаемых доходностей и стандартных отклонений
def find_E_n_sigma(data, tickers):
    expected_returns = {}
    risks = {}

    for ticker in tickers:
        risk = data[ticker].std()
        expected_returns[ticker] = data[ticker].mean()
        risks[ticker] = risk

    risk_and_return = pd.DataFrame({
        'Ticker': expected_returns.keys(),
        'E': expected_returns.values(),
        'σ': risks.values()
    })

    return risk_and_return

risk_and_return = find_E_n_sigma(data, tickers)

# Построение карты активов с выделенными выбранными.
plt.figure(figsize=(11, 9))
plt.scatter(risk_and_return['σ'], risk_and_return['E'], s=10, color='grey')
plt.title('«Карта» активов в системе координат (σ, E)')
plt.xlabel('Риск (σ)')
plt.ylabel('Ожидаемая доходность (E)')
plt.grid()
# plt.show()

''' =========  № 2  =========================================================================================================== '''
'''
###      Парето-оптимальные активы.
'''
''' =========================================================================================================================== '''

# Функция для поиска Парето-оптимальных активов
pareto_optimal_assets = []

for i in range(len(risk_and_return['Ticker'])):
    current_E = risk_and_return['E'][i]
    current_Sigma = risk_and_return['σ'][i]
    is_optimal = True
    for j in range(len(risk_and_return['Ticker'])):
        if i != j:
            if (risk_and_return['E'][j] >= current_E and risk_and_return['σ'][j] <= current_Sigma):
                is_optimal = False
                break
    if is_optimal:
        pareto_optimal_assets.append(risk_and_return['Ticker'][i])

pareto_optimal = find_E_n_sigma(data, pareto_optimal_assets)

''' =========  № 3  =========================================================================================================== '''
'''      Value at Risk и Conditional Value at Risk
###
###      Оценка VaR / CVaR с уровнем доверия 0,95 для Парето-оптимальных активов рынка.
###      Какие из активов наиболее предпочтительны по этим характеристикам?
###      Где они расположены на карте активов?
###      Сравнить результаты VaR и CVaR        
'''      
''' =========================================================================================================================== '''

def make_returns(portfolio):
    portfolio_returns = pd.DataFrame()
    ticker_returns = pd.DataFrame()

    for ticker in portfolio['Ticker']:
        ticker_returns['log_return'] = pd.DataFrame(data[ticker]).reset_index(drop=True)
        portfolio_returns[ticker] = ticker_returns['log_return']
    portfolio_returns['Portfolio Returns'] = portfolio_returns.sum(axis=1)
    
    return portfolio_returns['Portfolio Returns']

def historicalVaR(returns, alpha):

    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)

    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVaR, alpha=alpha)

    else:
        raise TypeError("Expected returns to be dataframe or series")

def historicalCVaR(returns, alpha):

    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()

    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVaR, alpha=alpha)

    else:
        raise TypeError("Expected returns to be dataframe or series")