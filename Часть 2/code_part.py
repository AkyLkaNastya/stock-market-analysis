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

# ''' =========  № 1  =========================================================================================================== '''
# '''
# '''
# '''
# '''
# ''' =========================================================================================================================== '''

# # Находим парето-оптимальные активы.
# pareto_optimal_assets = []

# assets = list(E_dict.keys())

# E_values = np.array(list(E_dict.values()))
# Sigma_values = np.array(list(Sigma_dict.values()))

# for i in range(len(assets)):
#     current_E = E_values[i]
#     current_Sigma = Sigma_values[i]
#     is_optimal = True
#     for j in range(len(assets)):
#         if i != j:
#             if (E_values[j] >= current_E and Sigma_values[j] <= current_Sigma):
#                 is_optimal = False
#                 break
#     if is_optimal:
#         pareto_optimal_assets.append(assets[i])

# # В нашем случае их получилось 35, а так как нам необходимо выбрать 50 активов, то нам надо найти ещё 15.
# to_find = 15

# '''_______________________________________________________________________________________

# Рассчитываем Z-оценку для стандартных отклонений log_returns для каждого актива.
# И фильтруем активы с  Z-оценкой, превышающими заданный порог (в данном случае 3), чтобы уменьшить влияние выбросов.
# _______________________________________________________________________________________'''

# z_scores = np.abs(stats.zscore(Sigma_values))
# threshold = 3
# filtered_Sigma_values = Sigma_values[(z_scores < threshold)]
# filtered_E_values = E_values[(z_scores < threshold)]
# filtered_assets = np.array(assets)[(z_scores < threshold)]

# correlation_matrix = data.pivot_table(values='log_return', index='Date', columns='Ticker').corr()

# selected_tickers = []

# selected_tickers.extend(pareto_optimal_assets)

# '''_______________________________________________________________________________________

# Цель этих циклов - обеспечить отбор достаточного количества активов,
# начиная с тех, которые наименее коррелируют с Парето-оптимальными активами,
# а затем, при необходимости, с теми, которые наиболее коррелируют.

# Этот подход направлен на диверсификацию ассортимента
# при одновременном обеспечении положительной ожидаемой доходности выбранных активов.
# ______________________________________________________________________________________'''


# added_assets = 0
# for asset in pareto_optimal_assets:
#     correlations = correlation_matrix[asset]
#     sorted_correlations = correlations.sort_values(ascending=True)
#     for ticker in sorted_correlations.index:
#         if ticker not in selected_tickers and ticker != asset and E_dict[ticker] > 0:
#             selected_tickers.append(ticker)
#             added_assets += 1
#             if added_assets == to_find:
#                 break
#     if added_assets == to_find:
#         break

# if added_assets < to_find:
#     for asset in pareto_optimal_assets:
#         correlations = correlation_matrix[asset]
#         sorted_correlations = correlations.sort_values(ascending=False)
#         for ticker in sorted_correlations.index:
#             if ticker not in selected_tickers and ticker != asset and E_dict[ticker] > 0:
#                 selected_tickers.append(ticker)
#                 added_assets += 1
#                 if added_assets == to_find:
#                     break
#         if added_assets == to_find:
#             break

# # Тикеры компаний, входящих в набор из 50 активов:
selected_tickers = ['ACNB34.SA', 'AGRO3.SA', 'ATSA11.SA', 'BAHI3.SA', 'BCRI11.SA',
                      'BRAX11.SA', 'COCA34.SA', 'CPFE3.SA', 'DHER34.SA', 'ENGI11.SA',
                      'EQMA3B.SA', 'ESUT11.SA', 'FISC11.SA', 'FLRP11.SA', 'FRIO3.SA',
                      'HFOF11.SA', 'ITUB3.SA', 'IVVB11.SA', 'KNHY11.SA', 'KNIP11.SA',
                      'MACY34.SA', 'MATB11.SA', 'MDTC34.SA', 'MRCK34.SA', 'MRSA3B.SA',
                      'PATI4.SA', 'PFIZ34.SA', 'RBRR11.SA', 'ROST34.SA', 'SUZB3.SA',
                      'TGAR11.SA', 'UNIP3.SA', 'UPAC34.SA', 'VRTA11.SA', 'WTSP11.SA',
                      'REDE3.SA', 'CRPG5.SA', 'CLSC4.SA', 'SNSY5.SA', 'EQPA6.SA',
                      'RSUL4.SA', 'MSPA4.SA', 'CLSC3.SA', 'DOHL4.SA', 'BSLI4.SA',
                      'EALT4.SA', 'BALM4.SA', 'BAUH4.SA', 'CTSA4.SA', 'CEED4.SA']

def find_E_n_sigma(data, tickers):
    expected_returns = {}
    risks = {}

    for ticker in tickers:
        risk = data[data['Ticker'] == ticker]['log_return'].std()
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

print(selected_risk_and_return)

# Построение карты активов с выделением выбранных.

# plt.figure(figsize=(11, 9))
# plt.scatter(risk_and_return['σ'], risk_and_return['E'])
# plt.scatter(selected_risk_and_return['σ'], selected_risk_and_return['E'], color='red')
# plt.title('«Карта» активов в системе координат (σ, E)')
# plt.xlabel('Риск (σ)')
# plt.ylabel('Ожидаемая доходность (E)')
# plt.grid()
# plt.show()


n_min_risk_assets = 10
weights = np.zeros(len(selected_tickers))
returns = np.zeros(len(selected_tickers))

'''_______________________________________________________________________________________

Цикл проходится по каждому выбранному тикеру, присваивая равную долю каждому активу в портфеле,
что дает равновзвешенный портфель.
_______________________________________________________________________________________'''

for i, ticker in enumerate(selected_tickers):
    weights[i] = 1 / len(selected_tickers)
    returns[i] = E_dict[ticker]

portfolio_df = pd.DataFrame({'Tickers': selected_tickers})
portfolio_df['Sigma'] = portfolio_df['Tickers'].map(Sigma_dict)
portfolio_df['Sigma_squared'] = [x*x for x in portfolio_df['Sigma']]

def portfolio_variance(weights_):
    cov_matrix = portfolio_df[['Sigma_squared']].to_numpy()
    portfolio_variance_ = weights_.T @ cov_matrix
    portfolio_variance_ = np.sum(portfolio_variance_ * weights_)
    return portfolio_variance_

'''_______________________________________________________________________________________

Ограничение гарантирует, что сумма долей равна 1,
что позволяет осуществлять короткие продажи. При этом доли могут быть отрицательными).
_______________________________________________________________________________________'''

constraints_with_short_sales = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
)

'''_______________________________________________________________________________________

constraints_without_short_sales также гарантирует, что сумма долей равна 1,
но добавляет ограничение неравенства, чтобы гарантировать, что все доли неотрицательны, запрещая короткие продажи.
_______________________________________________________________________________________'''

constraints_without_short_sales = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    {'type': 'ineq', 'fun': lambda x: x},
)

# Вычисляем оптимальный портфель, где короткие продажи запрещены.
result_without_short_sales = minimize(portfolio_variance, weights, method='SLSQP', constraints=constraints_without_short_sales)
optimal_weights = result_without_short_sales.x
top_10_assets_no_short_sales = np.argsort(optimal_weights)[-n_min_risk_assets:]

# Построение карты активов с минимальным риском (короткие продажи запрещены).

# plt.figure(figsize=(10, 6))
# for i in top_10_assets_no_short_sales:
#     plt.scatter(E_dict[selected_tickers[i]], Sigma_dict[selected_tickers[i]], color='orange', s=100)
#     plt.annotate(selected_tickers[i], (E_dict[selected_tickers[i]], Sigma_dict[selected_tickers[i]]), fontsize=8)
# plt.scatter(E_dict['KNIP11.SA'], Sigma_dict['KNIP11.SA'], color='orange',
#             s=100)
# plt.xlabel('Риск (σ)')
# plt.ylabel('Ожидаемая доходность (E)')
# plt.title('Активы с минимальным риском (короткие продажи запрещены)')
# plt.grid(True)
# plt.show()

# Вычисляем оптимальный портфель, где короткие продажи разрешены.
result_with_short_sales = minimize(portfolio_variance, weights, method='SLSQP', constraints=constraints_with_short_sales)
optimal_weights = result_with_short_sales.x
top_10_assets_no_short_sales = np.argsort(optimal_weights)[-n_min_risk_assets:]


# Построение карты активов с минимальным риском (короткие продажи разрешены).

# plt.figure(figsize=(10, 6))
# for i in top_10_assets_no_short_sales:
#     plt.scatter(E_dict[selected_tickers[i]], Sigma_dict[selected_tickers[i]], color='orange', s=100)
#     plt.annotate(selected_tickers[i], (E_dict[selected_tickers[i]], Sigma_dict[selected_tickers[i]]), fontsize=8)
# plt.scatter(E_dict['FRIO3.SA'], Sigma_dict['FRIO3.SA'], color='orange',
#             s=100)
# plt.xlabel('Риск (σ)')
# plt.ylabel('Ожидаемая доходность (E)')
# plt.title('Активы с минимальным риском (короткие продажи разрешены)')
# plt.grid(True)
# plt.show()


''' =========  № 4  =========================================================================================================== '''
'''
'''
'''
'''
''' =========================================================================================================================== '''

risk_aversion = 3

def objective_function(weights):
    portfolio_sigma = np.dot(weights, selected_risk_and_return['σ'])
    portfolio_return = np.dot(weights, selected_risk_and_return['E'])
    result = risk_aversion * portfolio_sigma - portfolio_return
    return result

# Оптимизация путем перебора различных вариантов долей в портфеле
def optimization(bound, constraints_):
    initial_weights = np.array([1/len(selected_tickers)] * len(selected_tickers))

    bounds = [(bound, None) for _ in range(len(selected_tickers))]
    
    result = minimize(objective_function, initial_weights, method='SLSQP', constraints=constraints_, bounds=bounds)
    optimal_weights = result.x

    portfolio = pd.DataFrame({'Ticker': selected_tickers, 'Weight': optimal_weights})
    portfolio = portfolio[portfolio['Weight'] != 0]

    return portfolio

# Считаем количество акций коротких и длительных продаж
def count_sales(portfolio):
    long_sales = 0
    short_sales = 0

    for i in portfolio['Weight']:
        if i > 0:
            long_sales += 1
        else:
            short_sales += 1
    
    result = f'''
    ==============================
    Длительные продажи: {long_sales} акций
    Короткие продажи: {short_sales} акций
    ==============================
    '''
    return result


portfolio_with_short_sales = optimization(None, constraints_with_short_sales)
#portfolio_without_short_sales = optimization(0, constraints_without_short_sales)