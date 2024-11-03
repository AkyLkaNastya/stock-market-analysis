import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize

pd.set_option('display.max_rows',None)

data = pd.read_csv("data_log_return.csv")
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

''' =========  № 1  =========================================================================================================== '''
'''
'''
'''
'''
''' =========================================================================================================================== '''

risk_and_return = find_E_n_sigma(data, tickers)
pareto_optimal_assets = find_pareto_optimal(risk_and_return)
pareto_optimal = find_E_n_sigma(data, pareto_optimal_assets)

def select_tickers(selected_tickers, mx):
    for ticker in tickers:
        returns = data[ticker].dropna() # Удаляем NaN, если они есть
        lst = data[ticker].tolist() # Создаем список лог. доходностей тикера

        # Проверяем тикер на соответствие
        if returns.shape[0] == 245 and lst.count(0) <= mx and risk_and_return.loc[risk_and_return['Ticker'] == ticker]['E'].values[0] > 0 and ticker not in selected_tickers:
            selected_tickers.append(ticker)

        if len(selected_tickers) >= 50: # Если мы набрали нужное кол-во тикеров, то прерываем функцию
            return selected_tickers
        
    return selected_tickers
        
selected_tickers = pareto_optimal_assets
mx = 1 

while len(selected_tickers) < 50:
    selected_tickers = select_tickers(selected_tickers, mx)
    mx += 1

selected_risk_and_return = find_E_n_sigma(data, selected_tickers)

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

# Define the objective function
def objective_function(weights):
    portfolio_sigma = np.dot(weights, selected_risk_and_return['σ'])
    return portfolio_sigma

# Calculate the minimum risk portfolio with short sales
initial_weights = np.array([1/len(selected_tickers)] * len(selected_tickers))
bounds = [(None, None) for _ in range(len(selected_tickers))]

result = minimize(objective_function, initial_weights, method='SLSQP', constraints=constraints_with_short_sales, bounds=bounds)
min_risk_portfolio_short_sales = pd.DataFrame({'Ticker': selected_tickers, 'Weight': result.x})

# Calculate the minimum risk portfolio without short sales
result = minimize(objective_function, initial_weights, method='SLSQP', constraints=constraints_without_short_sales, bounds=bounds)
min_risk_portfolio_no_short_sales = pd.DataFrame({'Ticker': selected_tickers, 'Weight': result.x})

# Calculate the expected return and standard deviation for the minimum risk portfolios
min_risk_portfolio_short_sales_return = np.dot(min_risk_portfolio_short_sales['Weight'], selected_risk_and_return['E'])
min_risk_portfolio_short_sales_sigma = np.dot(min_risk_portfolio_short_sales['Weight'], selected_risk_and_return['σ'])

min_risk_portfolio_no_short_sales_return = np.dot(min_risk_portfolio_no_short_sales['Weight'], selected_risk_and_return['E'])
min_risk_portfolio_no_short_sales_sigma = np.dot(min_risk_portfolio_no_short_sales['Weight'], selected_risk_and_return['σ'])

# Plot the portfolios on the (σ, E) coordinate system
plt.scatter(selected_risk_and_return['σ'], selected_risk_and_return['E'], label='Individual Assets')
plt.scatter(min_risk_portfolio_short_sales_sigma, min_risk_portfolio_short_sales_return, marker='*', color='red', label='Minimum Risk Portfolio (Short Sales)')
plt.scatter(min_risk_portfolio_no_short_sales_sigma, min_risk_portfolio_no_short_sales_return, marker='*', color='green', label='Minimum Risk Portfolio (No Short Sales)')
plt.xlabel('Standard Deviation (σ)')
plt.ylabel('Expected Return (E)')
plt.legend()
plt.show()

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