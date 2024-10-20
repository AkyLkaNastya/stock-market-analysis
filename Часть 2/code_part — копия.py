import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize

#pd.set_option('display.max_rows',None)

data = pd.read_csv('dateslog.csv')
tickers = list(data.columns)
tickers.pop(0)

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

# Функция для поиска Парето-оптимальных активов

def find_pareto_optimal(portfolio):
    pareto_optimal_assets = []

    assets = list(portfolio['Ticker'])
    portfolio_ = find_E_n_sigma(data, assets)

    E_values = np.array(list(portfolio_['E']))
    Sigma_values = np.array(list(portfolio_['σ']))

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

    return pareto_optimal_assets

risk_and_return = find_E_n_sigma(data, tickers)

''' =========  № 1  =========================================================================================================== '''
'''
'''
'''
'''
''' =========================================================================================================================== '''

pareto_optimal_assets = find_pareto_optimal(risk_and_return)

# В нашем случае их получилось 35, а так как нам необходимо выбрать 50 активов, то нам надо найти ещё 15.
to_find = 50 - len(pareto_optimal_assets)

'''_______________________________________________________________________________________

Рассчитываем Z-оценку для стандартных отклонений log_returns для каждого актива.
И фильтруем активы с  Z-оценкой, превышающими заданный порог (в данном случае 3), чтобы уменьшить влияние выбросов.
_______________________________________________________________________________________'''

E_dict = {}
Sigma_dict = {}

for ticker in tickers:
    risk = data[ticker].std()
    E_dict[ticker] = data[ticker].mean()
    Sigma_dict[ticker] = risk

E_values = np.array(list(E_dict.values()))
Sigma_values = np.array(list(Sigma_dict.values()))

z_scores = np.abs(stats.zscore(Sigma_values))
threshold = 3
filtered_Sigma_values = Sigma_values[(z_scores < threshold)]
filtered_E_values = E_values[(z_scores < threshold)]
filtered_assets = np.array(tickers)[(z_scores < threshold)]

correlation_matrix = data.pivot_table(index='Date').corr()

selected_tickers = []

selected_tickers.extend(pareto_optimal_assets)

'''_______________________________________________________________________________________

Цель этих циклов - обеспечить отбор достаточного количества активов,
начиная с тех, которые наименее коррелируют с Парето-оптимальными активами,
а затем, при необходимости, с теми, которые наиболее коррелируют.

Этот подход направлен на диверсификацию ассортимента
при одновременном обеспечении положительной ожидаемой доходности выбранных активов.
______________________________________________________________________________________'''


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

# # Тикеры компаний, входящих в набор из 50 активов:
selected_tickers_ = ['ARNC34.SA', 'ATSA11.SA', 'AZEV3.SA', 'CEGR3.SA', 'CPTS11.SA',
                     'FISC11.SA', 'HETA3.SA', 'HFOF11.SA', 'IRBR3.SA', 'LILY34.SA',
                     'MDTC34.SA', 'MRCK34.SA', 'PATI4.SA', 'RMAI11.SA', 'TGAR11.SA',
                     'VRTA11.SA', 'VVCR11.SA', 'WTSP11.SA', 'MERC4.SA', 'RAPT4.SA',
                     'SANB4.SA', 'PLAS3.SA', 'CRPG3.SA', 'OUJP11.SA', 'RNDP11.SA',
                     'BNBR3.SA', 'BIIB34.SA', 'SANB11.SA', 'ENGI11.SA', 'MOVI3.SA',
                     'DISB34.SA', 'DEAI34.SA', 'CEED3.SA', 'KNIP11.SA', 'POMO4.SA',
                     'CGAS3.SA', 'TEND3.SA', 'TRVC34.SA', 'JSRE11.SA', 'MOSC34.SA',
                     'DUKB34.SA', 'SPTW11.SA', 'RAIL3.SA', 'CMIG3.SA', 'MAXR11.SA',
                     'CMCS34.SA', 'ITLC34.SA', 'AMER3.SA', 'CEBR5.SA', 'MULT3.SA']

selected_risk_and_return = find_E_n_sigma(data, selected_tickers)

# Построение карты активов с выделенными выбранными.

# plt.figure(figsize=(11, 9))
# plt.scatter(risk_and_return['σ'], risk_and_return['E'], s=10, color='grey', label='Активы')
# plt.scatter(selected_risk_and_return['σ'], selected_risk_and_return['E'], s=10, color='red', label='Отобранные активы')
# plt.title('«Карта» активов в системе координат (σ, E)')
# plt.xlabel('Риск (σ)')
# plt.ylabel('Ожидаемая доходность (E)')
# plt.legend()
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
#     plt.scatter(Sigma_dict[selected_tickers[i]], E_dict[selected_tickers[i]], color='green', s=100)
#     plt.annotate(selected_tickers[i], (Sigma_dict[selected_tickers[i]], E_dict[selected_tickers[i]]), fontsize=8)
# plt.xlabel('Риск (σ)')
# plt.ylabel('Ожидаемая доходность (E)')
# plt.title('Активы с минимальным риском (короткие продажи запрещены)')
# plt.grid()

# Вычисляем оптимальный портфель, где короткие продажи разрешены.
result_with_short_sales = minimize(portfolio_variance, weights, method='SLSQP', constraints=constraints_with_short_sales)
optimal_weights = result_with_short_sales.x
top_10_assets_short_sales = np.argsort(optimal_weights)[-n_min_risk_assets:]

# Построение карты активов с минимальным риском (короткие продажи разрешены).

# plt.figure(figsize=(10, 6))
# for i in top_10_assets_short_sales:
#     plt.scatter(Sigma_dict[selected_tickers[i]], E_dict[selected_tickers[i]], color='orange', s=100)
#     plt.annotate(selected_tickers[i], (Sigma_dict[selected_tickers[i]], E_dict[selected_tickers[i]]), fontsize=8)
# plt.xlabel('Риск (σ)')
# plt.ylabel('Ожидаемая доходность (E)')
# plt.title('Активы с минимальным риском (короткие продажи разрешены)')
# plt.grid()

# Построение карты активов с минимальным риском (оба портфеля).

# plt.figure(figsize=(10, 6))
# for i in top_10_assets_no_short_sales:
#     plt.scatter(Sigma_dict[selected_tickers[i]], E_dict[selected_tickers[i]], color='green', s=100)
#     plt.annotate(selected_tickers[i], (Sigma_dict[selected_tickers[i]], E_dict[selected_tickers[i]]), fontsize=8)
# for i in top_10_assets_short_sales:
#     plt.scatter(Sigma_dict[selected_tickers[i]], E_dict[selected_tickers[i]], color='orange', s=40)
#     plt.annotate(selected_tickers[i], (Sigma_dict[selected_tickers[i]], E_dict[selected_tickers[i]]), fontsize=8)
# plt.xlabel('Риск (σ)')
# plt.ylabel('Ожидаемая доходность (E)')
# plt.title('Карта активов с минимальным риском')
# plt.grid()
# plt.show()

''' =========  № 2  =========================================================================================================== '''
'''
'''
'''
'''
''' =========================================================================================================================== '''



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


# portfolio_with_short_sales = optimization(None, constraints_with_short_sales).reset_index(drop=True)
# portfolio_without_short_sales = optimization(0, constraints_without_short_sales).reset_index(drop=True)

# def calculate_efficient_frontier(portfolio):
#     # Sort the portfolio by expected return in ascending order
#     portfolio = portfolio.sort_values(by='E')

#     # Initialize variables
#     expected_returns = []
#     standard_deviations = []

#     # Find the asset with the lowest risk (smallest σ)
#     lowest_risk_asset = portfolio.iloc[0]
#     lowest_risk = lowest_risk_asset['σ']
#     highest_return = lowest_risk_asset['E']

#     # Append the lowest risk asset to the lists
#     expected_returns.append(highest_return)
#     standard_deviations.append(lowest_risk)

#     # Iterate over different levels of risk (standard deviation)
#     for risk in np.arange(lowest_risk + 0.01, np.max(portfolio['σ']), 0.01):
#         # Find the portfolio with the highest expected return for the given risk level
#         assets = portfolio[portfolio['σ'] <= risk]
#         if not assets.empty:
#             portfolio_return = assets.iloc[-1]['E']
#             portfolio_risk = assets.iloc[-1]['σ']

#             # Append the portfolio return and risk to the lists
#             expected_returns.append(portfolio_return)
#             standard_deviations.append(portfolio_risk)

#     # Return the efficient frontier as a DataFrame
#     efficient_frontier = pd.DataFrame({
#         'σ': standard_deviations,
#         'E': expected_returns
#     })

#     # Plot the efficient frontier
#     plt.figure(figsize=(8, 6))
#     plt.scatter(portfolio['σ'], portfolio['E'], s=5)
#     plt.plot(efficient_frontier['σ'], efficient_frontier['E'], color='red')
#     plt.title('Efficient Frontier')
#     plt.xlabel('Risk (σ)')
#     plt.ylabel('Expected Return (E)')
#     plt.grid()
#     plt.show()

# portfolio_short_sales = find_E_n_sigma(data, portfolio_with_short_sales['Ticker'])
# portfolio_short_sales.insert(3, 'Weight', portfolio_with_short_sales['Weight'])

# calculate_efficient_frontier(portfolio_short_sales)

# portfolio_no_short_sales = find_E_n_sigma(data, portfolio_without_short_sales['Ticker'])
# portfolio_no_short_sales.insert(3, 'Weight', portfolio_without_short_sales['Weight'])

# calculate_efficient_frontier(portfolio_no_short_sales)''