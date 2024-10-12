import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('stock_data_2018.csv')
valid_tickers = data['Ticker'].unique().tolist()
selected_tickers = ['ACNB34.SA', 'AGRO3.SA', 'ATSA11.SA', 'BAHI3.SA', 'BALM4.SA',
                    'BAUH4.SA', 'BCRI11.SA', 'BOVV11.SA', 'BRAX11.SA', 'BSLI4.SA',
                    'CLSC3.SA', 'CLSC4.SA', 'COCA34.SA', 'CPFE3.SA', 'CRPG5.SA',
                    'CTSA4.SA', 'DHER34.SA', 'DOHL4.SA', 'EALT4.SA', 'ENGI11.SA',
                    'EQMA3B.SA', 'EQPA6.SA', 'ESUT11.SA', 'FISC11.SA', 'FLRP11.SA',
                    'FRIO3.SA', 'HFOF11.SA', 'ITUB3.SA', 'IVVB11.SA', 'KNHY11.SA',
                    'KNIP11.SA', 'MACY34.SA', 'MATB11.SA', 'MDTC34.SA', 'MRCK34.SA',
                    'MRSA3B.SA', 'MSPA4.SA', 'PATI4.SA', 'PFIZ34.SA', 'RBRR11.SA',
                    'REDE3.SA', 'ROST34.SA', 'RSUL4.SA', 'SNSY5.SA', 'SUZB3.SA',
                    'TGAR11.SA', 'UNIP3.SA', 'UPAC34.SA', 'VRTA11.SA', 'WTSP11.SA']

def making(data, tickers):
    E = {}
    Sigma = {}

    for ticker in tickers:
        log_returns = data[data['Ticker'] == ticker]['log_return']
        l = log_returns.std()
        if l < 1:
            E[ticker] = log_returns.mean()
            Sigma[ticker] = l

    risk_and_return = pd.DataFrame({
        'Ticker': E.keys(),
        'E': E.values(),
        'σ': Sigma.values()
    })

    return risk_and_return

risk_and_return = making(data, valid_tickers)

#   ------  TO DO  ----------------------------------------------------------
#
#   Разобраться почему в filtered_risk_and_return 48, а не 50 активов
#
#   -------------------------------------------------------------------------


filtered_risk_and_return = making(data[data['Ticker'].isin(selected_tickers)], selected_tickers)

plt.figure(figsize=(9, 7))
plt.scatter(risk_and_return['σ'], risk_and_return['E'])
plt.scatter(filtered_risk_and_return['σ'], filtered_risk_and_return['E'], color='red')
plt.title('«Карта» активов в системе координат (σ, E)')
plt.xlabel('Риск (σ)')
plt.ylabel('Ожидаемая доходность (E)')
plt.grid()
plt.show()