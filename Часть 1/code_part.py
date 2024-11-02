import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
import seaborn as sns

pd.set_option('display.max_rows',None)

data = pd.read_csv("data_log_return.csv")
tickers = list(data.columns)
tickers.pop(0)

''' =========  № 1  =========================================================================================================== '''
'''
###      Оценка ожидаемых доходностей и стандартных отклонений.
'''
''' =========================================================================================================================== '''

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
# plt.figure(figsize=(11, 9))
# plt.scatter(risk_and_return['σ'], risk_and_return['E'], s=10, color='grey')
# plt.title('«Карта» активов в системе координат (σ, E)')
# plt.xlabel('Риск (σ)')
# plt.ylabel('Ожидаемая доходность (E)')
# plt.grid()
# plt.show()

''' =========  № 2  =========================================================================================================== '''
'''
###      Парето-оптимальные активы.
'''
''' =========================================================================================================================== '''

print('='*20, 'Парето-оптимальные активы', '='*20)

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

print(pareto_optimal)

''' =========  № 3  =========================================================================================================== '''
'''      Value at Risk и Conditional Value at Risk
###
###      Оценка VaR / CVaR с уровнем доверия 0,95 для Парето-оптимальных активов рынка.
###      Какие из активов наиболее предпочтительны по этим характеристикам?
###      Где они расположены на карте активов?
###      Сравнить результаты VaR и CVaR        
'''      
''' =========================================================================================================================== '''

print('='*10, 'Оценка VaR и CVaR для Парето-оптимальных активов', '='*10)

confidence_level = 95

def monte_carlo_simulation(returns, num_simulations=10000):
    mean = returns.mean()
    std_dev = returns.std()
    simulated_returns = np.random.normal(mean, std_dev, (num_simulations, len(returns)))
    return simulated_returns

def calculate_historical_var_cvar(returns, confidence_level):
    var = -np.percentile(returns, 100 - confidence_level)
    losses = -returns
    var_threshold = np.percentile(losses, confidence_level)
    cvar = losses[losses >= var_threshold].mean()
    return var, cvar

def calculate_parametric_var_cvar(returns, confidence_level):
    mean = returns.mean()
    std_dev = returns.std()
    var = -norm.ppf(1 - confidence_level / 100) * std_dev + mean
    cvar = -mean + std_dev * norm.pdf(norm.ppf(confidence_level / 100)) / (1 - confidence_level / 100)
    return var, cvar

def calculate_monte_carlo_var_cvar(simulated_returns, confidence_level):
    var = -np.percentile(simulated_returns, 100 - confidence_level, axis=0)
    losses = -simulated_returns
    var_threshold = np.percentile(losses, confidence_level, axis=0)
    cvar = losses[losses >= var_threshold].mean(axis=0)
    return var, cvar

historical_method = pd.DataFrame()
historical_method['Ticker'] = pareto_optimal_assets
historical_method = historical_method.set_index('Ticker')
historical_method['VaR'] = np.nan
historical_method['CVaR'] = np.nan

parametric_method = pd.DataFrame()
parametric_method['Ticker'] = pareto_optimal_assets
parametric_method = parametric_method.set_index('Ticker')
parametric_method['VaR'] = np.nan
parametric_method['CVaR'] = np.nan

monte_carlo_method = pd.DataFrame()
monte_carlo_method['Ticker'] = pareto_optimal_assets
monte_carlo_method = monte_carlo_method.set_index('Ticker')
monte_carlo_method['VaR'] = np.nan
monte_carlo_method['CVaR'] = np.nan

for ticker in pareto_optimal_assets:
    log_returns = data[ticker].dropna()

    var, cvar = calculate_historical_var_cvar(log_returns, confidence_level)
    if var == 0:
        historical_method.loc[ticker, 'VaR'] = 0
    else:
        historical_method.loc[ticker, 'VaR'] = var
    historical_method.loc[ticker, 'CVaR'] = cvar

    var, cvar = calculate_parametric_var_cvar(log_returns, confidence_level)
    parametric_method.loc[ticker, 'VaR'] = var
    parametric_method.loc[ticker, 'CVaR'] = cvar

    simulated_returns = monte_carlo_simulation(log_returns)
    var, cvar = calculate_monte_carlo_var_cvar(simulated_returns, confidence_level)
    if isinstance(var, np.ndarray):
        var = var.mean()
    if isinstance(cvar, np.ndarray):
        cvar = cvar.mean()
    monte_carlo_method.loc[ticker, 'VaR'] = var
    monte_carlo_method.loc[ticker, 'CVaR'] = cvar

print('==== Исторический метод =====')
print(historical_method.sort_values('VaR').sort_values('CVaR'))

print('\n=== Параметрический метод ===')
print(parametric_method.sort_values('VaR').sort_values('CVaR'))

print('\n===== Метод Монте-Карло =====')
print(monte_carlo_method.sort_values('VaR').sort_values('CVaR'))

''' =========  № 4  =========================================================================================================== '''
'''      Нормальность распределений доходностей
###
###      В предположении, что наблюдаемые доходности выбранных активов являются повторной выборкой из некоторого распределения исследовать (выборочно) распределения доходностей выбранных активов.
###      Можно ли считать, что распределения доходностей подчиняются нормальному закону распределения?
###      Если ответ отрицательный, какие другие законы распределения доходностей соответствуют данным наблюдений?
'''      
''' =========================================================================================================================== '''

selected_tickers = []

for ticker in tickers:
    returns = data[ticker].dropna() # Удаляем NaN, если они есть
    lst = data[ticker].tolist()
    if returns.shape[0] == 245 and lst.count(0) <= 1 and risk_and_return.loc[risk_and_return['Ticker'] == ticker]['E'].values[0] > 0:
        selected_tickers.append(ticker)
    if len(selected_tickers) >= 5:
        break

for ticker in selected_tickers:
    returns = data[ticker].dropna() # Удаляем NaN, если они есть

    print('='*10, ticker, '='*70, '\n')

    # Тест Шапиро-Уилка
    stat, p_value = stats.shapiro(returns)
    print(f'Статистика = {stat}, p-значение = {p_value}')
    if p_value > 0.05:
        print(f'Распределение логарифмических доходностей можно считать нормальным (p > 0.05) \n')
    else:
        print(f'Распределение логарифмических доходностей не является нормальным (p <= 0.05) \n')


    # plt.figure(figsize=(7, 3))
    # plt.subplot(1, 2, 1)
    # sns.histplot(returns, bins=50, kde=True)
    # plt.title(f'Гистограмма: {ticker}')
    # plt.xlabel('Логарифмическая доходность')
    # plt.ylabel('Частота')

    # # QQ plot
    # plt.subplot(1, 2, 2)
    # stats.probplot(returns, dist="norm", plot=plt)
    # plt.title(f'QQ Plot')
    # plt.xlabel('')
    # plt.ylabel('')

    # # Показ графиков
    # plt.tight_layout()
    # plt.show()