from IPython import get_ipython;
get_ipython().magic('reset -sf')

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

# folders names definitions
root_folder = "/home/pandrade/use-case-rojo/"
data_folder = "01-Data/"
raw_data_folder = "01-Raw datasets/"
input_data = "input_data.xlsx"

# files names definitions
contratos_data_hora_file = "01(b)-Contratos I8 Data-Hora-McK.xlsx"
client_info_file = "02-Clients database (credit model).xlsx"
share_of_wallet_file = "03-Share of wallet (PEC dataset).xlsx"
market_US_Govt_Bond_10Y_file = "04.1-MarketData-US-Govt-Bond-10Y-hourly.xlsx"
market_US_Govt_Bond_5Y_file = "04.2-MarketData-US-Govt-Bond-5Y-hourly.xlsx"
market_Brazil_Bond_10Y_file = "04.3-MarketData-Brazil-Govt-Bond-10Y-daily.xlsx"
market_Brazil_Bond_5Y_file = "04.4-MarketData-Brazil-Govt-Bond-5Y-daily.xlsx"
market_JPMEMCI_file = "04.5-MarketData-JPMEMCI-hourly.xlsx"
market_MSCI_file = "04.6-MarketData-MSCI-hourly.xlsx"
market_B3_USD_Futures_file = "04.7-MarketData-B3-USD-Futures-hourly.xlsx"
market_BRL_file = "04.8-MarketData-BRL-hourly.xlsx"
market_MX_Peso_file = "04.9-MarketData-MX-Peso-hourly.xlsx"
market_CASADO_file = "04.10-MarketData-Casado_Bid_Ask.xlsx"
numero_contratos_file = "05-Numero_de_Contratos.xlsx"
market_CASADO_file = "04.10-MarketData-Casado_Bid_Ask.xlsx"
final_dataset_file = "01-FXTrades_ClientInfo_ShareOfWallet_MarketData.xlsx"
final_dataset_file_csv = "01-FXTrades_ClientInfo_ShareOfWallet_MarketData.csv"
final_dataset_file_csv = "01-FXTrades_ClientInfo_ShareOfWallet_MarketData_newdata.csv"

########################
dataset =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + input_data,
                  sheet_name= "data")
dataset['HORA_LOCAL'] = dataset['Hora'].astype(str).apply(lambda x: x[0:2])

# remover trades sem numero de boleto
dataset = dataset.loc[~pd.isnull(dataset['NUM_BOLETO'])]
# filtrar apenas corporate
dataset.loc[(dataset['Nome Plataforma'].apply(lambda x : x[:4])=='CORP') &
            (dataset['Segmento']=='ITAUBBA'),['Segmento']] = 'Corporate'
dataset = dataset.loc[dataset['Segmento'] == 'Corporate']
# filtrar apenas compra e venda
dataset.loc[np.isin(dataset['Produto'], ['Compra', 'Fin. Compra', 'Exp. Pronta', 'Simplex']), 'Produto'] = 'COMPRA'
dataset.loc[np.isin(dataset['Produto'], ['Venda' , 'Fin. Venda' , 'Imp. Pronta']), 'Produto'] = 'VENDA'
dataset = dataset.loc[np.isin(dataset.Produto, ['VENDA', 'COMPRA'])]
dataset.loc[dataset['Produto']=='COMPRA','Produto'] = 0
dataset.loc[dataset['Produto']=='VENDA','Produto'] = 1

# variavel corretora
dataset['Corretora'] = dataset['Corretora'].fillna(0)
dataset.loc[dataset['Corretora']!=0,'Corretora'] = 1
dataset = dataset.rename(columns = {'PRZU_MN':'PRAZO_REAL',
                                    'PRZU_ME':'PRAZO_USD',
                                    'vl_dol':'Volume USD',
                                    'Receitas':'Margem USD',
                                    'Cod Grupo Economico MCK':'Cod Grupo McK'})
# selectionar apenas USD
dataset = dataset.loc[dataset['paridade'] == 1]
# remover boletos duplicados
dataset = dataset.loc[~dataset['NUM_BOLETO'].duplicated(),:]
# filtrar apenas trades fechados
dataset = dataset.loc[dataset['Situacao']=='FECHADA',:]

# criar variavel: numero de transacoes anteriores
operacoes_anteriores =\
    dataset.loc[dataset['Situacao'] == 'FECHADA',['Data', 'Cod Grupo McK']]
operacoes_anteriores.dropna(inplace=True)

anteriores_1 = operacoes_anteriores.rename(columns={'Data':'Data1'})
anteriores_2 = operacoes_anteriores.rename(columns={'Data':'Data2'})

operacoes_anteriores_cross =\
    pd.merge(anteriores_1,
             anteriores_2,
             how='inner',
             left_on=['Cod Grupo McK'],
             right_on = ['Cod Grupo McK'])

anteriores_30 = \
    operacoes_anteriores_cross.loc[
        (operacoes_anteriores_cross['Data1'] >operacoes_anteriores_cross['Data2']) &
        ((operacoes_anteriores_cross['Data1']-operacoes_anteriores_cross['Data2'])<=pd.Timedelta(30,'D')),:]
anteriores_60 = \
    operacoes_anteriores_cross.loc[
        (operacoes_anteriores_cross['Data1'] >operacoes_anteriores_cross['Data2']) &
        (operacoes_anteriores_cross['Data1']-operacoes_anteriores_cross['Data2']<=pd.Timedelta(60,'D')),:]
anteriores_90 = \
    operacoes_anteriores_cross.loc[
        (operacoes_anteriores_cross['Data1'] >operacoes_anteriores_cross['Data2']) &
        (operacoes_anteriores_cross['Data1']-operacoes_anteriores_cross['Data2']<=pd.Timedelta(90,'D')),:]

anteriores_30 = anteriores_30[['Cod Grupo McK', 'Data1']].groupby(['Cod Grupo McK','Data1']).size().reset_index(name='counts')
anteriores_30.rename(columns={'Data1':'Data','counts':'Numero_Operacoes_30D'}, inplace=True)
anteriores_60 = anteriores_60[['Cod Grupo McK', 'Data1']].groupby(['Cod Grupo McK','Data1']).size().reset_index(name='counts')
anteriores_60.rename(columns={'Data1':'Data','counts':'Numero_Operacoes_60D'}, inplace=True)
anteriores_90 = anteriores_90[['Cod Grupo McK', 'Data1']].groupby(['Cod Grupo McK','Data1']).size().reset_index(name='counts')
anteriores_90.rename(columns={'Data1':'Data','counts':'Numero_Operacoes_90D'}, inplace=True)

dataset = pd.merge(dataset, anteriores_30,  how='left',
                  left_on=['Cod Grupo McK','Data'],
                  right_on = ['Cod Grupo McK','Data'])
dataset = pd.merge(dataset, anteriores_60,  how='left',
                  left_on=['Cod Grupo McK','Data'],
                  right_on = ['Cod Grupo McK','Data'])
dataset = pd.merge(dataset, anteriores_90,  how='left',
                  left_on=['Cod Grupo McK','Data'],
                  right_on = ['Cod Grupo McK','Data'])

# selecionar epnas trades a partir de 2016
dataset = dataset.loc[dataset['Data'] >= datetime.strptime('2016-01-01', '%Y-%m-%d')]

dataset[['Numero_Operacoes_30D',
         'Numero_Operacoes_60D',
         'Numero_Operacoes_90D']] =\
    dataset[['Numero_Operacoes_30D',
              'Numero_Operacoes_60D',
              'Numero_Operacoes_90D']].fillna(value = 0)

# criar variavel com data e hora
dataset['DateHourTrade'] = pd.to_datetime(dataset['Data'].astype(str) + " " + dataset['Hora'].astype(str))
# criar variavel com ID do trade
dataset['trade_id'] = range(dataset.shape[0])
# descartar colunas desnecessarias
dataset.drop(columns = ['paridade',
                        'Segmento',
                        'Plataforma',
                        'Nome Plataforma',
                        'Moeda',
                        'Situacao',
                        'Valor ME',
                        'PrazoDC',
                        'PRZC_MN',
                        'PRZC_ME',
                        'Regiao'], inplace=True)
dataset['safra_trade'] = dataset['Data'].astype(str).apply(lambda x: x[0:4]) + dataset['Data'].astype(str).apply(lambda x: x[5:7])
print(dataset.shape)
print(len(dataset['Cod Grupo McK'].unique()))
########################
## Client info (credit model) dataset
########################
client_info =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + client_info_file,
                  sheet_name= "data")
dataset = dataset.loc[np.isin(dataset['Cod Grupo McK'], client_info['ID Group Mck'].unique())]

print(dataset.shape)
print(len(dataset['Cod Grupo McK'].unique()))

d1 = dataset[['trade_id','Data','Hora', 'Cod Grupo McK','Produto']]
d2 = dataset[['trade_id','Data','Hora', 'Cod Grupo McK','Produto']]
d1.rename(columns={'trade_id':'trade_id1',
                    'Hora':'hora1',
                   'Produto':'Produto1'}, inplace=True)
d2.rename(columns={'trade_id':'trade_id2',
                    'Hora':'hora2',
                   'Produto':'Produto2'}, inplace=True)
d1d2 = pd.merge(d1, d2,  how='inner',
                left_on=['Cod Grupo McK','Data'],
                right_on = ['Cod Grupo McK','Data'])
d1d2 = d1d2.loc[d1d2['trade_id1'] != d1d2['trade_id2']]
d1d2 = d1d2.loc[pd.to_datetime(d1d2['Data'].astype(str) + " " + d1d2['hora1'].astype(str))>
                pd.to_datetime(d1d2['Data'].astype(str) + " " + d1d2['hora2'].astype(str))]
d1d2['time_diff'] = pd.to_datetime(d1d2['Data'].astype(str) + " " + d1d2['hora1'].astype(str))-\
                    pd.to_datetime(d1d2['Data'].astype(str) + " " + d1d2['hora2'].astype(str))

d1d2 = d1d2.loc[d1d2['Produto1']!=d1d2['Produto2']]
d1d2 = d1d2.loc[d1d2['time_diff'].apply(lambda x: x.total_seconds())<(5*60)]

d1d2.drop(columns=['hora1','hora2','time_diff','Produto1', 'Produto2'], inplace = True)
d1d2['new_trade_id'] = range(0, d1d2.shape[0])

d1d2 = d1d2.melt(id_vars=['Cod Grupo McK','Data','new_trade_id']).rename(columns = {'value':'trade_id'}).drop(
    columns = ['variable','Cod Grupo McK', 'Data'])
d1d2['current_trade_id'] = d1d2['trade_id']
d1d2.sort_values(by=['trade_id'], inplace=True)
while sum(d1d2.groupby('current_trade_id')['new_trade_id'].nunique()>1)>0:
    grouped_trade_id = pd.DataFrame(d1d2.groupby('current_trade_id')['new_trade_id'].min()).reset_index()
    d1d2 = d1d2.rename(columns={'new_trade_id':'new_trade_id_1'})
    d1d2 = pd.merge(d1d2, grouped_trade_id, how='inner',
                    left_on=['current_trade_id'],
                    right_on=['current_trade_id'])
    d1d2 = d1d2.drop(columns=['current_trade_id'])
    d1d2 = d1d2.rename(columns={'new_trade_id_1':'current_trade_id'})

d1d2 = pd.DataFrame(d1d2.groupby(['trade_id','new_trade_id']).size()).reset_index()
d1d2 = pd.merge(d1d2,dataset,how='inner',left_on=['trade_id'],right_on=['trade_id'])

d1d2.sort_values(by=['DateHourTrade'], inplace=True)
d1d2 = d1d2.groupby(['Cod Grupo McK','Produto','Data', 'new_trade_id']).agg(
        {'trade_id': lambda x: x.iloc[0],
         'Hora': 'min',
         'Margem USD':'sum',
         'Volume USD':'sum',
         'Taxa Base': lambda x: x.iloc[0],
         'Taxa Contrato': lambda x: x.iloc[0],
         'PRAZO_REAL': lambda x: x.iloc[0],
         'PRAZO_USD': lambda x: x.iloc[0],
         'DateHourTrade': lambda x: x.iloc[0],
         'safra_trade': lambda x: x.iloc[0]}).reset_index()
d1d2.drop(columns = ['new_trade_id'], inplace = True)
dataset = dataset.append(d1d2, sort=False)
print(dataset.shape)

dataset.sort_values(by=['Cod Grupo McK', 'Data', 'Produto'], inplace=True)
dataset['Transacoes_Anteriores_Dia'] = 1
transacoes_dia = \
    pd.DataFrame(
        dataset.groupby(
            by=['Cod Grupo McK', 'Data', 'Produto', 'trade_id'])['Transacoes_Anteriores_Dia'].sum().\
            groupby(by=['Cod Grupo McK', 'Data', 'Produto']).cumsum().reset_index())
transacoes_dia['Transacoes_Anteriores_Dia']=transacoes_dia['Transacoes_Anteriores_Dia']-1
dataset = dataset.drop(columns = ['Transacoes_Anteriores_Dia'])
dataset = pd.merge(dataset, transacoes_dia,  how='inner',
                left_on=['Cod Grupo McK', 'Data', 'Produto', 'trade_id'],
                right_on = ['Cod Grupo McK', 'Data', 'Produto', 'trade_id'])
print(dataset.shape)
dataset = dataset.loc[dataset['Volume USD'] >= 100]
print(dataset.shape)
########################
### client info
########################

client_info['Year_cli_info'] = client_info['Safra'].astype(str).apply(lambda x: int(x[0:4]))
client_info['Month_cli_info'] = client_info['Safra'].astype(str).apply(lambda x: x[4:6])
client_info['Day_cli_info'] = "01"
client_info.loc[client_info['Month_cli_info'] == '12', 'Year_cli_info'] =\
    client_info.loc[client_info['Month_cli_info'] == '12', 'Year_cli_info'].apply(lambda x: x+1)
client_info['Month_cli_info'] = client_info['Month_cli_info'].astype(str).apply(lambda x: int(x))%12+1
client_info['Month_cli_info'] = client_info['Month_cli_info'].apply(lambda x: format(x, '02'))
client_info['clnfo_datehour'] = \
    pd.to_datetime(client_info['Year_cli_info'].astype(str) + "-" +
                   client_info['Month_cli_info'] + "-" +
                   client_info['Day_cli_info'])-timedelta(seconds=1)

dataset = dataset.set_index('Cod Grupo McK').join(client_info.set_index('ID Group Mck'), how= 'inner')
dataset1 = dataset.loc[dataset['DateHourTrade'] > dataset['clnfo_datehour'],:]
dataset1_id_clihour = pd.DataFrame(dataset1.groupby(['trade_id'], sort=False)['clnfo_datehour'].max())
dataset1_id_clihour = dataset1_id_clihour.reset_index()

dataset2 = dataset.loc[~dataset['trade_id'].isin(dataset1_id_clihour['trade_id']),:]
dataset2_id_clihour = pd.DataFrame(dataset2.groupby(['trade_id'], sort=False)['clnfo_datehour'].min())
dataset2_id_clihour = dataset2_id_clihour.reset_index()

dataset['Cod Grupo McK'] = dataset.index

dataset1 = pd.merge(dataset,
                    dataset1_id_clihour,
                    how='inner',
                    left_on=['trade_id','clnfo_datehour'],
                    right_on = ['trade_id','clnfo_datehour'])

dataset2 = pd.merge(dataset,
                    dataset2_id_clihour,
                    how='inner',
                    left_on=['trade_id','clnfo_datehour'],
                    right_on = ['trade_id','clnfo_datehour'])
dataset = dataset1.append(dataset2)

print("Dataset shape and number of clients:")
print(dataset.shape)
print(len(dataset['Cod Grupo McK'].unique()))

########################
# Share of wallet (PEC dataset)
share_of_wallet = \
    pd.read_excel(root_folder + data_folder + raw_data_folder + share_of_wallet_file,
                  sheet_name= "data")

share_of_wallet.rename(columns = {'YEAR':'YEAR_ShareOfWallet', 'MONTH':'MONTH_ShareOfWallet'},
                       inplace=True)

share_of_wallet['Safra_ShareOfWallet'] =\
    share_of_wallet['YEAR_ShareOfWallet'].astype(str) +\
    share_of_wallet['MONTH_ShareOfWallet'].astype(str)
share_of_wallet['Safra_ShareOfWallet'] =\
    share_of_wallet['Safra_ShareOfWallet'].astype(int)

dataset = dataset.set_index('Cod Grupo McK').join(share_of_wallet.set_index('COD GRUPO MCK'), how= 'inner')
dataset1 = dataset.loc[dataset['safra_trade'].astype(int) > dataset['Safra_ShareOfWallet'],:]
dataset1_id_sfra_shareow = pd.DataFrame(dataset1.groupby(['trade_id'], sort=False)['Safra_ShareOfWallet'].max())
dataset1_id_sfra_shareow = dataset1_id_sfra_shareow.reset_index()

dataset2 = dataset.loc[~dataset['trade_id'].isin(dataset1_id_sfra_shareow['trade_id']),:]
dataset2_id_sfra_shareow = pd.DataFrame(dataset2.groupby(['trade_id'], sort=False)['Safra_ShareOfWallet'].min())
dataset2_id_sfra_shareow = dataset2_id_sfra_shareow.reset_index()

dataset['Cod Grupo McK'] = dataset.index

dataset1 = pd.merge(dataset,
                    dataset1_id_sfra_shareow,
                    how='inner',
                    left_on=['trade_id','Safra_ShareOfWallet'],
                    right_on = ['trade_id','Safra_ShareOfWallet'])

dataset2 = pd.merge(dataset,
                    dataset2_id_sfra_shareow,
                    how='inner',
                    left_on=['trade_id','Safra_ShareOfWallet'],
                    right_on = ['trade_id','Safra_ShareOfWallet'])
dataset = dataset1.append(dataset2)

print(dataset.shape)
print(len(dataset['Cod Grupo McK'].unique()))

#####
# Market - US Govt Bond 10Y database
US_Govt_Bond_10Y =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + market_US_Govt_Bond_10Y_file,
                  sheet_name= "data")
dataset = dataset.set_index("DateHourTrade").join(US_Govt_Bond_10Y.set_index("Date"), how="outer")
dataset.sort_index(inplace =True)
c_list2 = ["Open", "High", "Low", "Close"]
dataset[c_list2] = \
    dataset[c_list2].fillna(method="ffill")
dataset.rename( columns={"Open":"US_Govt_Bond_10Y_Open",
                         "High":"US_Govt_Bond_10Y_High",
                         "Low":"US_Govt_Bond_10Y_Low",
                         "Close":"US_Govt_Bond_10Y_Close"}, inplace=True)
dataset.dropna(subset=[
                  'Data',
                  'US_Govt_Bond_10Y_Open',
                  'US_Govt_Bond_10Y_High',
                  'US_Govt_Bond_10Y_Low',
                  'US_Govt_Bond_10Y_Close'])
print("Dataset shape and number of clients:")
print(dataset.shape)
print(len(dataset['Cod Grupo McK'].unique()))

#####
# Market - US Govt Bond 5Y database
US_Govt_Bond_5Y =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + market_US_Govt_Bond_5Y_file,
                  sheet_name= "data")
dataset = dataset.join(US_Govt_Bond_5Y.set_index("Date"), how="outer")
dataset.sort_index(inplace =True)
c_list2 = ["Open", "High", "Low", "Close"]
dataset[c_list2] = \
    dataset[c_list2].fillna(method="ffill")
# dataset[c_list2] = \
#     dataset[c_list2].fillna(method="bfill")
dataset.rename( columns={"Open":"US_Govt_Bond_5Y_Open",
                         "High":"US_Govt_Bond_5Y_High",
                         "Low":"US_Govt_Bond_5Y_Low",
                         "Close":"US_Govt_Bond_5Y_Close"}, inplace=True)
dataset.dropna(subset=[
                  'Data',
                  'US_Govt_Bond_5Y_Open',
                  'US_Govt_Bond_5Y_High',
                  'US_Govt_Bond_5Y_Low',
                  'US_Govt_Bond_5Y_Close'])

print(dataset.shape)
print(len(dataset['Cod Grupo McK'].unique()))

#####
# Market - Brazil Govt Bond 10Y database
Brazil_Govt_Bond_10Y =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + market_Brazil_Bond_10Y_file,
                  sheet_name= "data")
dataset = dataset.join(Brazil_Govt_Bond_10Y.set_index("Date"), how="outer")
dataset.sort_index(inplace =True)
c_list2 = ["Open", "High", "Low", "Close"]
dataset[c_list2] = \
    dataset[c_list2].fillna(method="ffill")
dataset.rename( columns={"Open":"Brazil_Bond_10Y_Open",
                         "High":"Brazil_Bond_10Y_High",
                         "Low":"Brazil_Bond_10Y_Low",
                         "Close":"Brazil_Bond_10Y_Close"}, inplace=True)
dataset.dropna(subset=[
                  'Data',
                  'Brazil_Bond_10Y_Open',
                  'Brazil_Bond_10Y_High',
                  'Brazil_Bond_10Y_Low',
                  'Brazil_Bond_10Y_Close'])

print("Dataset shape and number of clients:")
print(dataset.shape)
print(len(dataset['Cod Grupo McK'].unique()))

#####
# Market - Brazil Govt Bond 5Y database
Brazil_Govt_Bond_5Y =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + market_Brazil_Bond_5Y_file,
                  sheet_name= "data")
dataset = dataset.join(Brazil_Govt_Bond_5Y.set_index("Date"), how="outer")
dataset.sort_index(inplace =True)
c_list2 = ["Open", "High", "Low", "Close"]
dataset[c_list2] = \
    dataset[c_list2].fillna(method="ffill")
# dataset[c_list2] = \
#     dataset[c_list2].fillna(method="bfill")
dataset.rename(columns={"Open":"Brazil_Bond_5Y_Open",
                         "High":"Brazil_Bond_5Y_High",
                         "Low":"Brazil_Bond_5Y_Low",
                         "Close":"Brazil_Bond_5Y_Close"}, inplace=True)
dataset.dropna(subset=[
                  'Data',
                  'Brazil_Bond_5Y_Open',
                  'Brazil_Bond_5Y_High',
                  'Brazil_Bond_5Y_Low',
                  'Brazil_Bond_5Y_Close'])
print(dataset.shape)
print(len(dataset['Cod Grupo McK'].unique()))

#####
# JPM Emerging Market currency index
jpmemci =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + market_JPMEMCI_file,
                  sheet_name="data")
dataset = dataset.join(jpmemci.set_index("Date"), how="outer")
dataset.sort_index(inplace =True)
c_list1 = ["90 Days", "60 Days", "30 Days","15 Days", "1 Day", "12 Hours", "6 Hours", "4 Hours", "2 Hours"]
for c in c_list1:
    dataset.loc[dataset[c].astype(str) == "A", c] = np.nan
c_list2 = ["Open", "High", "Low", "Close", "90 Days", "60 Days", "30 Days",
                "15 Days", "1 Day", "12 Hours", "6 Hours", "4 Hours", "2 Hours"]
dataset[c_list2] = \
    dataset[c_list2].fillna(method="ffill")
dataset.rename( columns={"Open":"JPMEMCI_Open","High":"JPMEMCI_High","Low":"JPMEMCI_Low","Close":"JPMEMCI_Close",
                         "90 Days":"JPMEMCI_VOLAT_90_Days", "60 Days":"JPMEMCI_VOLAT_60_Days", "30 Days":"JPMEMCI_VOLAT_30_Days",
                         "15 Days":"JPMEMCI_VOLAT_15_Days", "1 Day":"JPMEMCI_VOLAT_1_Day", "12 Hours":"JPMEMCI_VOLAT_12_Hours",
                         "6 Hours":"JPMEMCI_VOLAT_6_Hours", "4 Hours":"JPMEMCI_VOLAT_4_Hours", "2 Hours":"JPMEMCI_VOLAT_2_Hours"}, inplace=True)
dataset.dropna(subset=[
         "Data",
         "JPMEMCI_VOLAT_90_Days",
         "JPMEMCI_VOLAT_60_Days",
         "JPMEMCI_VOLAT_30_Days",
         "JPMEMCI_VOLAT_15_Days",
         "JPMEMCI_VOLAT_1_Day",
         "JPMEMCI_VOLAT_12_Hours",
         "JPMEMCI_VOLAT_6_Hours",
         "JPMEMCI_VOLAT_4_Hours",
         "JPMEMCI_VOLAT_2_Hours"])
dataset[["JPMEMCI_VOLAT_90_Days",
         "JPMEMCI_VOLAT_60_Days",
         "JPMEMCI_VOLAT_30_Days",
         "JPMEMCI_VOLAT_15_Days",
         "JPMEMCI_VOLAT_1_Day",
         "JPMEMCI_VOLAT_12_Hours",
         "JPMEMCI_VOLAT_6_Hours",
         "JPMEMCI_VOLAT_4_Hours",
         "JPMEMCI_VOLAT_2_Hours"
         ]] = dataset[["JPMEMCI_VOLAT_90_Days",
                       "JPMEMCI_VOLAT_60_Days",
                       "JPMEMCI_VOLAT_30_Days",
                       "JPMEMCI_VOLAT_15_Days",
                       "JPMEMCI_VOLAT_1_Day",
                       "JPMEMCI_VOLAT_12_Hours",
                       "JPMEMCI_VOLAT_6_Hours",
                       "JPMEMCI_VOLAT_4_Hours",
                       "JPMEMCI_VOLAT_2_Hours"]].astype(float)

#####
# Morgan Stanley Capital International - Emerging Markets Currency Index
msci =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + market_MSCI_file,
                  sheet_name="data")
dataset = dataset.join(msci.set_index("Date"), how="outer")
dataset.sort_index(inplace =True)
for c in c_list1:
    dataset.loc[dataset[c].astype(str) == "A", c] = np.nan
dataset[c_list2] = \
    dataset[c_list2].fillna(method="ffill")
# dataset[c_list2] = \
#     dataset[c_list2].fillna(method="bfill")
dataset.rename( columns={"Open":"MSCI_Open","High":"MSCI_High","Low":"MSCI_Low","Close":"MSCI_Close",
                        "90 Days":"MSCI_VOLAT_90_Days", "60 Days":"MSCI_VOLAT_60_Days", "30 Days":"MSCI_VOLAT_30_Days",
                        "15 Days":"MSCI_VOLAT_15_Days", "1 Day":"MSCI_VOLAT_1_Day", "12 Hours":"MSCI_VOLAT_12_Hours",
                        "6 Hours":"MSCI_VOLAT_6_Hours", "4 Hours":"MSCI_VOLAT_4_Hours", "2 Hours":"MSCI_VOLAT_2_Hours"},
               inplace=True)
dataset.dropna(subset=[
         "Data",
         "MSCI_VOLAT_90_Days",
         "MSCI_VOLAT_60_Days",
         "MSCI_VOLAT_30_Days",
         "MSCI_VOLAT_15_Days",
         "MSCI_VOLAT_1_Day",
         "MSCI_VOLAT_12_Hours",
         "MSCI_VOLAT_6_Hours",
         "MSCI_VOLAT_4_Hours",
         "MSCI_VOLAT_2_Hours"])
dataset[["MSCI_VOLAT_90_Days",
         "MSCI_VOLAT_60_Days",
         "MSCI_VOLAT_30_Days",
         "MSCI_VOLAT_15_Days",
         "MSCI_VOLAT_1_Day",
         "MSCI_VOLAT_12_Hours",
         "MSCI_VOLAT_6_Hours",
         "MSCI_VOLAT_4_Hours",
         "MSCI_VOLAT_2_Hours"
         ]] = dataset[["MSCI_VOLAT_90_Days",
                       "MSCI_VOLAT_60_Days",
                       "MSCI_VOLAT_30_Days",
                       "MSCI_VOLAT_15_Days",
                       "MSCI_VOLAT_1_Day",
                       "MSCI_VOLAT_12_Hours",
                       "MSCI_VOLAT_6_Hours",
                       "MSCI_VOLAT_4_Hours",
                       "MSCI_VOLAT_2_Hours"]].astype(float)
#####
# B3 USD - primeiro futuro USD
b3usd_futures =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + market_B3_USD_Futures_file,
                  sheet_name="data")
dataset = dataset.join(b3usd_futures.set_index("Date"), how="outer")
dataset.sort_index(inplace =True)
for c in c_list1:
    dataset.loc[dataset[c].astype(str) == "A", c] = np.nan
dataset[c_list2] = \
    dataset[c_list2].fillna(method="ffill")
dataset.rename( columns={"Open":"B3USD_Open","High":"B3USD_High","Low":"B3USD_Low","Close":"B3USD_Close",
                        "90 Days":"B3USD_VOLAT_90_Days", "60 Days":"B3USD_VOLAT_60_Days", "30 Days":"B3USD_VOLAT_30_Days",
                        "15 Days":"B3USD_VOLAT_15_Days", "1 Day":"B3USD_VOLAT_1_Day", "12 Hours":"B3USD_VOLAT_12_Hours",
                        "6 Hours":"B3USD_VOLAT_6_Hours", "4 Hours":"B3USD_VOLAT_4_Hours", "2 Hours":"B3USD_VOLAT_2_Hours"},
               inplace=True)
dataset.dropna(subset=[
        "Data",
        "B3USD_VOLAT_90_Days",
         "B3USD_VOLAT_60_Days",
         "B3USD_VOLAT_30_Days",
         "B3USD_VOLAT_15_Days",
         "B3USD_VOLAT_1_Day",
         "B3USD_VOLAT_12_Hours",
         "B3USD_VOLAT_6_Hours",
         "B3USD_VOLAT_4_Hours",
         "B3USD_VOLAT_2_Hours"])
dataset[["B3USD_VOLAT_90_Days",
         "B3USD_VOLAT_60_Days",
         "B3USD_VOLAT_30_Days",
         "B3USD_VOLAT_15_Days",
         "B3USD_VOLAT_1_Day",
         "B3USD_VOLAT_12_Hours",
         "B3USD_VOLAT_6_Hours",
         "B3USD_VOLAT_4_Hours",
         "B3USD_VOLAT_2_Hours"
         ]] = dataset[["B3USD_VOLAT_90_Days",
                       "B3USD_VOLAT_60_Days",
                       "B3USD_VOLAT_30_Days",
                       "B3USD_VOLAT_15_Days",
                       "B3USD_VOLAT_1_Day",
                       "B3USD_VOLAT_12_Hours",
                       "B3USD_VOLAT_6_Hours",
                       "B3USD_VOLAT_4_Hours",
                       "B3USD_VOLAT_2_Hours"]].astype(float)
#####
# Brazilian dollar vs USD
brl =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + market_BRL_file,
                  sheet_name="data")
dataset = dataset.join(brl.set_index("Date"), how="outer")
dataset.sort_index(inplace =True)
for c in c_list1:
    dataset.loc[dataset[c].astype(str) == "A", c] = np.nan
dataset[c_list2] = \
    dataset[c_list2].fillna(method="ffill")
dataset.rename( columns={"Open":"BRL_Open","High":"BRL_High","Low":"BRL_Low","Close":"BRL_Close",
                        "90 Days":"BRL_VOLAT_90_Days", "60 Days":"BRL_VOLAT_60_Days", "30 Days":"BRL_VOLAT_30_Days",
                        "15 Days":"BRL_VOLAT_15_Days", "1 Day":"BRL_VOLAT_1_Day", "12 Hours":"BRL_VOLAT_12_Hours",
                        "6 Hours":"BRL_VOLAT_6_Hours", "4 Hours":"BRL_VOLAT_4_Hours", "2 Hours":"BRL_VOLAT_2_Hours"},
               inplace=True)
dataset.dropna(subset=[
        "Data",
         "BRL_VOLAT_90_Days",
         "BRL_VOLAT_60_Days",
         "BRL_VOLAT_30_Days",
         "BRL_VOLAT_15_Days",
         "BRL_VOLAT_1_Day",
         "BRL_VOLAT_12_Hours",
         "BRL_VOLAT_6_Hours",
         "BRL_VOLAT_4_Hours",
         "BRL_VOLAT_2_Hours"])
dataset[["BRL_VOLAT_90_Days",
         "BRL_VOLAT_60_Days",
         "BRL_VOLAT_30_Days",
         "BRL_VOLAT_15_Days",
         "BRL_VOLAT_1_Day",
         "BRL_VOLAT_12_Hours",
         "BRL_VOLAT_6_Hours",
         "BRL_VOLAT_4_Hours",
         "BRL_VOLAT_2_Hours"
         ]] = dataset[["BRL_VOLAT_90_Days",
                       "BRL_VOLAT_60_Days",
                       "BRL_VOLAT_30_Days",
                       "BRL_VOLAT_15_Days",
                       "BRL_VOLAT_1_Day",
                       "BRL_VOLAT_12_Hours",
                       "BRL_VOLAT_6_Hours",
                       "BRL_VOLAT_4_Hours",
                       "BRL_VOLAT_2_Hours"]].astype(float)
#####
# Mexican Peso vs USD
mx_peso =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + market_MX_Peso_file,
                  sheet_name="data")
dataset = dataset.join(mx_peso.set_index("Date"), how="outer")
dataset.sort_index(inplace =True)
for c in c_list1:
    dataset.loc[dataset[c].astype(str) == "A", c] = np.nan
dataset[c_list2] = \
    dataset[c_list2].fillna(method="ffill")
# dataset[c_list2] = \
#     dataset[c_list2].fillna(method="bfill")
dataset.rename( columns={"Open":"MX_PESO_Open","High":"MX_PESO_High","Low":"MX_PESO_Low","Close":"MX_PESO_Close",
                        "90 Days":"MX_PESO_VOLAT_90_Days", "60 Days":"MX_PESO_VOLAT_60_Days", "30 Days":"MX_PESO_VOLAT_30_Days",
                        "15 Days":"MX_PESO_VOLAT_15_Days", "1 Day":"MX_PESO_VOLAT_1_Day", "12 Hours":"MX_PESO_VOLAT_12_Hours",
                        "6 Hours":"MX_PESO_VOLAT_6_Hours", "4 Hours":"MX_PESO_VOLAT_4_Hours", "2 Hours":"MX_PESO_VOLAT_2_Hours"},
               inplace=True)
dataset.dropna(subset=[
         "Data",
         "MX_PESO_VOLAT_90_Days",
         "MX_PESO_VOLAT_60_Days",
         "MX_PESO_VOLAT_30_Days",
         "MX_PESO_VOLAT_15_Days",
         "MX_PESO_VOLAT_1_Day",
         "MX_PESO_VOLAT_12_Hours",
         "MX_PESO_VOLAT_6_Hours",
         "MX_PESO_VOLAT_4_Hours",
         "MX_PESO_VOLAT_2_Hours"])
dataset[["MX_PESO_VOLAT_90_Days",
         "MX_PESO_VOLAT_60_Days",
         "MX_PESO_VOLAT_30_Days",
         "MX_PESO_VOLAT_15_Days",
         "MX_PESO_VOLAT_1_Day",
         "MX_PESO_VOLAT_12_Hours",
         "MX_PESO_VOLAT_6_Hours",
         "MX_PESO_VOLAT_4_Hours",
         "MX_PESO_VOLAT_2_Hours"
         ]] = dataset[["MX_PESO_VOLAT_90_Days",
                       "MX_PESO_VOLAT_60_Days",
                       "MX_PESO_VOLAT_30_Days",
                       "MX_PESO_VOLAT_15_Days",
                       "MX_PESO_VOLAT_1_Day",
                       "MX_PESO_VOLAT_12_Hours",
                       "MX_PESO_VOLAT_6_Hours",
                       "MX_PESO_VOLAT_4_Hours",
                       "MX_PESO_VOLAT_2_Hours"]].astype(float)
#####
# CASADO Bloomberg Index
casado =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + market_CASADO_file,
                  sheet_name="data")
print(dataset.shape)
dataset = dataset.join(casado.set_index("DateHour"), how="outer")
print(dataset.shape)
dataset.sort_index(inplace =True)
dataset[['CASADO_Curncy_Bid_Price','CASADO_Curncy_Ask_Price']] = \
    dataset[['CASADO_Curncy_Bid_Price','CASADO_Curncy_Ask_Price']].fillna(method="ffill")
dataset = \
    dataset.dropna(subset=['Data','CASADO_Curncy_Bid_Price','CASADO_Curncy_Ask_Price'])
dataset[['CASADO_Curncy_Bid_Price','CASADO_Curncy_Ask_Price']] =\
    dataset[['CASADO_Curncy_Bid_Price','CASADO_Curncy_Ask_Price']].astype(float)
dataset['CASADO_Curncy_Bid_minus_Ask_Price'] =\
    dataset['CASADO_Curncy_Bid_Price'] - dataset['CASADO_Curncy_Ask_Price']
print(dataset.shape)

dataset = dataset.reset_index().rename(columns={'index':'datetime'})
###
# Add number of contracts
contratos =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + numero_contratos_file,
                  sheet_name="data")
print(contratos.shape)
contratos = contratos.groupby('IDE_BOLETO').max().reset_index()
print(contratos.shape)

print(dataset.shape)
dataset =\
    pd.merge(dataset,
             contratos,
             how='inner',
             left_on=['NUM_BOLETO'],
             right_on=['IDE_BOLETO'])
print(dataset.shape)
####
# Remove columns
dataset.drop(columns=['NUM_BOLETO',
                       'Hora',
                      'Data',
                      'Year_cli_info',
                      'Month_cli_info',
                      'Day_cli_info',
                      'clnfo_datehour',
                      'ID',
                      'YEAR_ShareOfWallet',
                      'MONTH_ShareOfWallet',
                      'Aba',
                      'safra_trade',
                      'Safra',
                      'Safra_ShareOfWallet',
                      'trade_id',
                      'Taxa Contrato',
                      'IDE_BOLETO'], inplace=True)

print(dataset.shape)
print(len(dataset['Cod Grupo McK'].unique()))

writer = pd.ExcelWriter(root_folder + data_folder + final_dataset_folder + final_dataset_file)
dataset.to_excel(writer,"data", index =False)
writer.save()
print(dataset.shape)
dataset.to_csv(root_folder + data_folder + final_dataset_folder + final_dataset_file_csv,
               sep =';',
               decimal =',',
               index =False)
print(dataset['datetime'])
