import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize

# Считываем данные из файла
data = pd.read_csv('stock_data_2018.csv')
tickers = data['Ticker'].unique()

# Вычисляем оценки ожидаемых доходностей и стандартных отклонений
E_dict = {}
Sigma_dict = {}

for ticker in tickers:
    risk = data[data['Ticker'] == ticker]['log_return'].std()
    E_dict[ticker] = data[data['Ticker'] == ticker]['log_return'].mean()
    Sigma_dict[ticker] = risk

# Находим парето-оптимальные активы.
pareto_optimal_assets = []

assets = list(E_dict.keys())

E_values = np.array(list(E_dict.values()))
Sigma_values = np.array(list(Sigma_dict.values()))

for i in range(len(assets)):
    current_E = E_values[i]
    current_Sigma = Sigma_values[i]
    is_optimal = True
    for j in range(len(assets)):
        if i != j:
            if (E_values[j] >= current_E and Sigma_values[j] <= current_Sigma):
                is_optimal = False
                break
    if is_optimal:
        pareto_optimal_assets.append(assets[i])

# В нашем случае их получилось 35, а так как нам необходимо выбрать 50 активов, то нам надо найти ещё 15.
to_find = 15

# Рассчитываем Z-оценку для стандартных отклонений log_returns для каждого актива.
# И фильтруем активы с  Z-оценкой, превышающими заданный порог (в данном случае 3), чтобы уменьшить влияние выбросов.

z_scores = np.abs(stats.zscore(Sigma_values))
threshold = 3
filtered_Sigma_values = Sigma_values[(z_scores < threshold)]
filtered_E_values = E_values[(z_scores < threshold)]
filtered_assets = np.array(assets)[(z_scores < threshold)]

correlation_matrix = data.pivot_table(values='log_return', index='Date', columns='Ticker').corr()

selected_tickers = []

selected_tickers.extend(pareto_optimal_assets)

# Цель этих циклов - обеспечить отбор достаточного количества активов,
# начиная с тех, которые наименее коррелируют с Парето-оптимальными активами,
# а затем, при необходимости, с теми, которые наиболее коррелируют.
#
# Этот подход направлен на диверсификацию ассортимента
# при одновременном обеспечении положительной ожидаемой доходности выбранных активов.

added_assets = 0
for asset in pareto_optimal_assets:
    correlations = correlation_matrix[asset]
    sorted_correlations = correlations.sort_values(ascending=True)
    for ticker in sorted_correlations.index:
        if ticker not in selected_tickers and ticker != asset and E_dict[ticker] > 0:
            selected_tickers.append(ticker)
            added_assets += 1
            if added_assets == to_find:
                break
    if added_assets == to_find:
        break

if added_assets < to_find:
    for asset in pareto_optimal_assets:
        correlations = correlation_matrix[asset]
        sorted_correlations = correlations.sort_values(ascending=False)
        for ticker in sorted_correlations.index:
            if ticker not in selected_tickers and ticker != asset and E_dict[ticker] > 0:
                selected_tickers.append(ticker)
                added_assets += 1
                if added_assets == to_find:
                    break
        if added_assets == to_find:
            break

# Тикеры компаний, входящих в набор из 50 активов:
#
# selected_tickers = ['ACNB34.SA', 'AGRO3.SA', 'ATSA11.SA', 'BAHI3.SA', 'BALM4.SA',
#                     'BAUH4.SA', 'BCRI11.SA', 'BOVV11.SA', 'BRAX11.SA', 'BSLI4.SA',
#                     'CLSC3.SA', 'CLSC4.SA', 'COCA34.SA', 'CPFE3.SA', 'CRPG5.SA',
#                     'CTSA4.SA', 'DHER34.SA', 'DOHL4.SA', 'EALT4.SA', 'ENGI11.SA',
#                     'EQMA3B.SA', 'EQPA6.SA', 'ESUT11.SA', 'FISC11.SA', 'FLRP11.SA',
#                     'FRIO3.SA', 'HFOF11.SA', 'ITUB3.SA', 'IVVB11.SA', 'KNHY11.SA',
#                     'KNIP11.SA', 'MACY34.SA', 'MATB11.SA', 'MDTC34.SA', 'MRCK34.SA',
#                     'MRSA3B.SA', 'MSPA4.SA', 'PATI4.SA', 'PFIZ34.SA', 'RBRR11.SA',
#                     'REDE3.SA', 'ROST34.SA', 'RSUL4.SA', 'SNSY5.SA', 'SUZB3.SA',
#                     'TGAR11.SA', 'UNIP3.SA', 'UPAC34.SA', 'VRTA11.SA', 'WTSP11.SA']

# Строим карту активов с отмеченными красным 50 активами.
'''
def find_E_n_sigma(data, tickers):
    expected_returns = {}
    risks = {}

    for ticker in tickers:
        risk = data[data['Ticker'] == ticker]['log_return'].std()
        if risk < 1: # Скрываем выбросы
            expected_returns[ticker] = data[data['Ticker'] == ticker]['log_return'].mean()
            risks[ticker] = risk

    risk_and_return = pd.DataFrame({
        'Ticker': expected_returns.keys(),
        'E': expected_returns.values(),
        'σ': risks.values()
    })

    return risk_and_return

risk_and_return = find_E_n_sigma(data, tickers)
selected_risk_and_return = find_E_n_sigma(data, selected_tickers)
3
plt.figure(figsize=(11, 9))
plt.scatter(risk_and_return['σ'], risk_and_return['E'])
plt.scatter(selected_risk_and_return['σ'], selected_risk_and_return['E'], color='green')
plt.title('«Карта» активов в системе координат (σ, E)')
plt.xlabel('Риск (σ)')
plt.ylabel('Ожидаемая доходность (E)')
plt.grid()
plt.show()
'''

# def portfolio_optimization(returns, cov_matrix, risk_free_rate=0.01, allow_short=True):
#     num_assets = len(returns)
#     args = (returns, cov_matrix, risk_free_rate)
    
#     def portfolio_performance(weights, returns, cov_matrix, risk_free_rate):
#         portfolio_return = np.sum(returns * weights)
#         portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
#         return portfolio_volatility, portfolio_return
    
#     def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
#         p_var, p_ret = portfolio_performance(weights, returns, cov_matrix, risk_free_rate)
#         return -(p_ret - risk_free_rate) / p_var
    
#     constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
#     bounds = None if allow_short else tuple((0, 1) for _ in range(num_assets))
    
#     result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
#                       method='SLSQP', bounds=bounds, constraints=constraints)
    
#     return result.x

# # Calculate expected returns and covariance matrix
# returns = selected_risk_and_return['E'].values
# cov_matrix = data.pivot_table(index='Date', columns='Ticker', values='log_return').cov().values

# # Optimize portfolio
# optimal_weights = portfolio_optimization(returns, cov_matrix, allow_short=True)
# print("Optimal Weights with Short Selling Allowed:", optimal_weights)

# def calculate_var_cvar(portfolio_returns, confidence_level=0.95):
#     sorted_returns = np.sort(portfolio_returns)
#     index = int((1 - confidence_level) * len(sorted_returns))
#     var = sorted_returns[index]
#     cvar = sorted_returns[:index].mean()
#     return var, cvar

# # Simulate portfolio returns
# portfolio_returns = np.dot(data.pivot_table(index='Date', columns='Ticker', values='log_return').values, optimal_weights)

# # Calculate VaR and CVaR
# var, cvar = calculate_var_cvar(portfolio_returns)
# print("VaR:", var)
# print("CVaR:", cvar)

# plt.figure(figsize=(11, 9))
# plt.scatter(selected_risk_and_return['σ'], selected_risk_and_return['E'], label='Assets')
# optimal_portfolio_return = np.dot(returns, optimal_weights)
# optimal_portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
# plt.scatter(optimal_portfolio_risk, optimal_portfolio_return, color='green', label='Optimal Portfolio')
# plt.title('Asset Map (σ, E)')
# plt.xlabel('Risk (σ)')
# plt.ylabel('Expected Return (E)')
# plt.legend()
# plt.grid()
# plt.show()

# '''
# Modify the portfolio_optimization function to set allow_short=False and repeat the steps.

# These steps will help you construct and analyze an optimal portfolio based on your risk tolerance
# and constraints. Adjust the code as needed to fit your specific data and requirements.
# '''