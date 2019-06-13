from IPython import get_ipython;
get_ipython().magic('reset -sf')

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
from sklearn.decomposition import PCA
import scipy
import xgboost as xgb
import random
from itertools import compress

# folders names definitions
root_folder = 'C:/Users/Pablo Andrade/Documents/2019-01-Jan/01-BTU-FX-Pricing/09-Hand-over/'
root_folder = 'C:/Users/Pablo Andrade/Box Sync/FX Pricing Model/09-Handover/'
data_folder = '01-Data/'
raw_data_folder = '01-Raw datasets/'
final_dataset_folder = '02-Final dataset/'
model_visualization_folder = '04-Models visualization/'
model_validation_folder = '05-Model validation/'
model_object_folder = '06-Models objects (pickle)/'

# files names definitions
final_dataset_file_csv = '01-FXTrades_ClientInfo_ShareOfWallet_MarketData.csv'
final_dataset_file_csv = "01-FXTrades_ClientInfo_ShareOfWallet_MarketData_newdata.csv"
features_importance_rf_csv = 'varImp-randomforest.csv'
features_importance_xgboost_csv = 'varImp-XGBoost.csv'
features_gain_xgboost_csv = 'varImp-gain-XGBoost.csv'
features_correlation_with_target_csv =  'features_correlatoin-with-target.csv'
xgboost_reference_price_model =  'xgboost_reference_price_model.pickle'
features_xgboost_file = 'features_xgboost_file.pickle'
features_xgboost_gain_file = 'features_xgboost_gain_file.pickle'
X_train_file = 'X_train.pickle'
X_test_file = 'X_test.pickle'
y_train_file = 'y_train.pickle'
y_test_file = 'y_test.pickle'
xgboost_mse_total_output_file = 'xgboost_mse_total_output_file.pickle'
xgboost_mse_var_output_file = 'xgboost_mse_var_output_file.pickle'
xgboost_mse_bias_output_file = 'xgboost_mse_bias_output_file.pickle'
xgboost_mse_bias_abs_output_file = 'xgboost_mse_bias_abs_output_file.pickle'

negative_spreads_file = "Check-Trades-Negative_spreads.xlsx"
over300_spreads_file = "Check-Trades-Spreads_over_300.xlsx"

plot_rf_predctionVSreal_file = 'RF_predction_vs_real.png'
plot_training_validation_file = 'XGBoost_training_vs_validation_error.png'
html_xgb_file1 = 'html_file1.html'

# read dataset file
input_data = pd.read_csv(root_folder + data_folder + final_dataset_folder + final_dataset_file_csv,
                         sep =';',
                         decimal =',')
#####
features_importance =\
    pd.read_csv(root_folder + model_validation_folder + features_importance_xgboost_csv, sep=';', decimal=',')
# create target variable (new variable of spread in base points)
# variable target
target_variable = ['Spread bps']
input_data['Spread bps'] = 10000 * (input_data['Margem USD'] / input_data['Volume USD'])
input_data = input_data.loc[input_data['Spread bps'] > 0]
input_data = input_data.loc[input_data['Spread bps'] < 300]
targets = input_data[target_variable]
datetime = input_data['datetime']
client_IDs = input_data['Cod Grupo McK']
# ID variables (variable for identification of user/date)
id_variables = ['Cod Grupo McK']
# variables duplicated (other variables have same meaning)
duplicated_variables = ['Margem USD','Rating_Grupo']
# variables to be removed (not necessary to model)
remove_variables = ['CAM57_Região', 'CAM57_Plat','Tipo Contabil','Regiao']
# variables with too many factors
variables_too_many_factors = ['Pais Origem Capital','Setor Economico', 'Setor 38','Setor 30 Raw']
# numierc variables
numeric_variables = ['Volume USD','Taxa Base','PRAZO_REAL', 'PRAZO_USD','Transacoes_Anteriores_Dia',
                     'SoC','Share SV Bradesco','Share SV BB','Share SV Santander','Share SV City','Share SV Caixa','Share SV Safra',
                     'Share SV Banri','Share SV Outros','IU Itau','IU Mercado','SOR','CV_prosp','CV_iCEAp_med_VP_Cred','RARoC_g_cred',
                     'RARoC_g','EAD_inicial','HURDLE','MFB_Cash','MFB_Credito','MFB_Fundos','MFB_IB','MFB_XSell','RGO_Cash','RGO_Credito',
                     'RGO_Fundos','RGO_IB','RGO_XSell','CUSTO_K_Cash','CUSTO_K_Credito','CUSTO_K_Fundos','CUSTO_K_IB','CUSTO_K_XSell',
                     'PB_Cash','PB_Credito','PB_Fundos','PB_IB','PB_XSell','MFB_Total','MFB_ratio','MFB_ratio_ex_IB','PB_Total','PB_ratio',
                     'PB_ratio_ex_IB','RiscoxLimite','Risco Cred','Vendas Liquidas','Tam Relativo ao Setor','Share SV Bancos Grandes',
                     'logPD', 'logPD_Media_2', 'logPD_Media_12','logPD_Tendencia','QTD_Assets','Notional_Assets',
                     'QTD_Assets_1', 'QTD_Assets_3', 'QTD_Assets_6','QTD_Assets_12', 'Notional_Assets_1', 'Notional_Assets_3',
                     'Notional_Assets_6', 'Notional_Assets_12', 'SoC_Media_2', 'SoC_Media_12', 'SoC_Tendencia', 'IU Itau_Media_2',
                     'IU Itau_Media_12', 'IU Itau_Tendencia', 'IU Mercado_Media_2','IU Mercado_Media_12', 'IU Mercado_Tendencia',
                     'CAM57_ACC', 'CAM57_Export','CAM57_Financeiro Compra', 'CAM57_Import', 'CAM57_Financeiro Venda',
                     'CAM57_Subtotal pronto', 'CAM57_Total', 'CAM57_ITAU-ACC','CAM57_ITAU-Compra',
                     'CAM57_ITAU-Compras Futuras', 'CAM57_ITAU-Venda','CAM57_ITAU-Vendas Futuras', 'CAM57_ITAU-Subtotal pronto',
                     'CAM57_ITAU-Total', 'CAM57_SHARE_ITAU-ACC', 'CAM57_SHARE_ITAU-Compra','CAM57_SHARE_ITAU-Venda',
                     'CAM57_SHARE_ITAU-Total','CAM57_Lost_Deals-ACC', 'CAM57_Lost_Deals-Compra','CAM57_Lost_Deals-Venda',
                     'US_Govt_Bond_10Y_Open', 'US_Govt_Bond_10Y_High','US_Govt_Bond_10Y_Low', 'US_Govt_Bond_10Y_Close',
                     'US_Govt_Bond_5Y_Open', 'US_Govt_Bond_5Y_High','US_Govt_Bond_5Y_Low', 'US_Govt_Bond_5Y_Close',
                     'Brazil_Bond_10Y_Open', 'Brazil_Bond_10Y_High','Brazil_Bond_10Y_Low', 'Brazil_Bond_10Y_Close',
                     'Brazil_Bond_5Y_Open', 'Brazil_Bond_5Y_High', 'Brazil_Bond_5Y_Low','Brazil_Bond_5Y_Close',
                     'Transacoes_Anteriores_Dia',
                     'JPMEMCI_Open', 'JPMEMCI_High','JPMEMCI_Low', 'JPMEMCI_Close', 'JPMEMCI_VOLAT_90_Days',
                     'JPMEMCI_VOLAT_60_Days', 'JPMEMCI_VOLAT_30_Days','JPMEMCI_VOLAT_15_Days', 'JPMEMCI_VOLAT_1_Day',
                     'JPMEMCI_VOLAT_12_Hours', 'JPMEMCI_VOLAT_6_Hours','JPMEMCI_VOLAT_4_Hours', 'JPMEMCI_VOLAT_2_Hours',
                     'MSCI_Open','MSCI_High', 'MSCI_Low', 'MSCI_Close', 'MSCI_VOLAT_90_Days','MSCI_VOLAT_60_Days',
                     'MSCI_VOLAT_30_Days', 'MSCI_VOLAT_15_Days','MSCI_VOLAT_1_Day', 'MSCI_VOLAT_12_Hours', 'MSCI_VOLAT_6_Hours',
                     'MSCI_VOLAT_4_Hours', 'MSCI_VOLAT_2_Hours', 'B3USD_Open',
                     'B3USD_High', 'B3USD_Low', 'B3USD_Close', 'B3USD_VOLAT_90_Days',
                     'B3USD_VOLAT_60_Days', 'B3USD_VOLAT_30_Days',
                     'B3USD_VOLAT_15_Days', 'B3USD_VOLAT_1_Day', 'B3USD_VOLAT_12_Hours',
                     'B3USD_VOLAT_6_Hours', 'B3USD_VOLAT_4_Hours','B3USD_VOLAT_2_Hours', 'BRL_Open', 'BRL_High',
                     'BRL_Low','BRL_Close', 'BRL_VOLAT_90_Days', 'BRL_VOLAT_60_Days',
                     'BRL_VOLAT_30_Days', 'BRL_VOLAT_15_Days', 'BRL_VOLAT_1_Day','BRL_VOLAT_12_Hours', 'BRL_VOLAT_6_Hours',
                     'BRL_VOLAT_4_Hours','BRL_VOLAT_2_Hours', 'MX_PESO_Open', 'MX_PESO_High', 'MX_PESO_Low',
                     'MX_PESO_Close', 'MX_PESO_VOLAT_90_Days', 'MX_PESO_VOLAT_60_Days',
                     'MX_PESO_VOLAT_30_Days', 'MX_PESO_VOLAT_15_Days','MX_PESO_VOLAT_1_Day', 'MX_PESO_VOLAT_12_Hours',
                     'MX_PESO_VOLAT_6_Hours', 'MX_PESO_VOLAT_4_Hours','MX_PESO_VOLAT_2_Hours']
# categorical variables
categorical_variables = ['Produto','Tipo Capital','Auditado Por','Fonte do Balanco','Auditoria Balanco','Setor 30',
                         'Relacionamento com Bradesco','Relacionamento com BB','Relacionamento com Santander',
                         'Relacionamento com City','Relacionamento com Caixa','Relacionamento com Safra',
                         'Relacionamento com Banri','Relacionamento com Outros','Relacionamento com Bradesco e BB',
                         'Relacionamento com Bancos Pequenos','Relacionamento com Bancos Grandes','MAX_Complexity',
                         'MAX_Complexity_1','MAX_Complexity_3','MAX_Complexity_6','MAX_Complexity_12']
#####
# remove variables
#####
# id variables
input_data = input_data.loc[:, ~ np.isin(input_data.columns, id_variables)]
# duplicated variables
input_data = input_data.loc[:, ~ np.isin(input_data.columns, duplicated_variables)]
# variables to be removed
input_data = input_data.loc[:, ~ np.isin(input_data.columns, remove_variables)]
# variables with too many factors
input_data = input_data.loc[:, ~ np.isin(input_data.columns, variables_too_many_factors)]
# target variable
input_data = input_data.loc[:, ~ np.isin(input_data.columns, target_variable)]
#####
# numeric variables
# #####
# PB_ratio and PB_ratio_ex_IB: string to numeric variables
input_data.loc[input_data['logPD'].isna(),'logPD'] = input_data['logPD'].min()
for variable in numeric_variables:
    input_data.loc[input_data[variable].isna(), variable] = input_data[variable].mean()
for variable in numeric_variables:
    input_data[variable] = input_data[variable].astype(float)
    if len(input_data.loc[input_data[variable].isna(),variable])>0 :
        print(variable)
        print(len(input_data.loc[input_data[variable].isna(),variable]))
#####
# categorical variables
#####
# trader: string to numeric variables (fillna with zero)
# input_data['Trader'] = input_data['Trader'].fillna(0).astype(int)
# relatioship variables
variables_relationship =\
    ['Relacionamento com Bradesco',
     'Relacionamento com BB',
     'Relacionamento com Santander',
     'Relacionamento com City',
     'Relacionamento com Caixa',
     'Relacionamento com Safra',
     'Relacionamento com Banri',
     'Relacionamento com Outros',
     'Relacionamento com Bradesco e BB',
     'Relacionamento com Bancos Pequenos',
     'Relacionamento com Bancos Grandes']
for variable in variables_relationship:
    input_data.loc[input_data[variable]==False,variable] = 0
    input_data.loc[input_data[variable]==True,variable] = 1
    input_data.loc[input_data[variable].isna(),variable] = 2
# complexity variables
complexity_relationship =\
    ['MAX_Complexity',
     'MAX_Complexity_1',
     'MAX_Complexity_3',
     'MAX_Complexity_6',
     'MAX_Complexity_12']
# complexity variables
for variable in complexity_relationship:
    input_data.loc[:,variable] = input_data.loc[:,variable].astype(int)
for variable in categorical_variables:
    if len(input_data.loc[input_data[variable].isna(),variable])>0 :
        print(variable)
        print(len(input_data.loc[input_data[variable].isna(),variable]))
# transform categorical variable using Label encoder
le = preprocessing.LabelEncoder()
for column in categorical_variables:
    print(column)
    input_data[column] = le.fit_transform(input_data[column].astype(str))
# Corretora
input_data.loc[input_data['Corretora'].isna(),'Corretora'] = 0
input_data.loc[input_data['Corretora']!=0,'Corretora'] = 1
input_data.loc[input_data['Numero_Operacoes_30D'].isna(),'Numero_Operacoes_30D'] = 0
input_data.loc[input_data['Numero_Operacoes_60D'].isna(),'Numero_Operacoes_60D'] = 0
input_data.loc[input_data['Numero_Operacoes_90D'].isna(),'Numero_Operacoes_90D'] = 0
####
# remove special characters
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
input_data.columns =\
    [regex.sub("_", col)
     if any(x in str(col)
            for x in set(('[', ']', '<')))
     else col for col in input_data.columns.values]
input_data = input_data.dropna(subset=features_importance['Feature'].unique())
#####
with open(root_folder + model_object_folder + xgboost_mse_total_output_file, "rb") as pickle_mse_total_output_file:
    mse_total_xgboost_estimator = pickle.load(pickle_mse_total_output_file)
with open(root_folder + model_object_folder + xgboost_mse_var_output_file, "rb") as pickle_mse_var_output_file:
    mse_var_xgboost_estimator = pickle.load(pickle_mse_var_output_file)
with open(root_folder + model_object_folder + xgboost_mse_bias_abs_output_file, "rb") as pickle_mse_bias_abs_sq_output_file:
    mse_bias_abs_sq_xgboost_estimator = pickle.load(pickle_mse_bias_abs_sq_output_file)
#####
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data, targets, test_size=0.10, random_state=42)
X_train = X_train.sort_values(by="datetime")
######
### use train dataset
n = 9
size = int(np.ceil(X_train.shape[0]/n))

# save the model to disk
with open(root_folder + model_object_folder + xgboost_reference_price_model, "rb") as pickle_output_file:
    xgb_model = pickle.load(pickle_output_file)

n_components_selected =5

scaler = preprocessing.StandardScaler()
pca = PCA(n_components=n_components_selected)

X_train_selected_vars = X_train.loc[:, np.isin(X_train.columns, features_importance['Feature'].unique())]
X_train_scaled = scaler.fit_transform(X_train_selected_vars)
####################
SEED = 4449
random.seed(SEED)
s = random.sample(range(0, X_train_scaled.shape[0]), 100)
sample1 = X_train_scaled[s,:]
spread1 = y_train['Spread bps'].iloc[s].values
s11 = random.sample(range(0, sample1.shape[0]), 60)
sample11 = sample1[s11,:]
spread11 = spread1[s11]
spread11.shape = (spread11.shape[0], 1)
spread12 = spread1[~np.isin(range(0, sample1.shape[0]), s11)]
sample12 = sample1[~np.isin(range(0, sample1.shape[0]), s11),:]
spread12.shape = (sample12.shape[0], 1)
col_correlation_spread = []
for column in range(0,X_train_scaled.shape[1]):
    col11 = sample11[:,column]
    col11.shape = (sample11.shape[0],1)
    col12 = sample12[:,column]
    col12.shape = (sample12.shape[0], 1)
    aux11 = np.zeros( (col11.shape[0],1))
    aux12 = np.zeros( (col12.shape[0],1))
    distances_col = scipy.spatial.distance.cdist(np.hstack((col11,aux11)),
                                                 np.hstack((col12,aux12)),
                                                 metric='euclidean')
    distances_spread = scipy.spatial.distance.cdist(np.hstack((spread11,aux11)),
                                                    np.hstack((spread12,aux12)),
                                                    metric='euclidean')
    ###
    distances_spread.shape = (1,distances_spread.shape[0]*distances_spread.shape[1])
    distances_col.shape = (1,distances_col.shape[0]*distances_col.shape[1])
    ###
    # plt.figure()
    # plt.plot(distances_col[0,:], distances_spread[0,:], '.')
    # plt.xlabel('spatial distance')
    # plt.ylabel('spread distance')
    # plt.title(X_train_selected_vars.columns[column])
    col_correlation_spread.append(np.corrcoef(distances_col[0,:], distances_spread[0,:])[0,1])
####################
#X_train_scaled = X_train_scaled[:,[x[0] for x in enumerate(col_correlation_spread) if x[1] > 0.0]]
X_train_scaled_pca = pca.fit_transform(X_train_scaled)
df_train_scaled_pca = pd.DataFrame(data=X_train_scaled_pca)
###

chunk = 0
X_train_chunk = df_train_scaled_pca.iloc[chunk:chunk + size, :]
X_train_xgb_chunk = X_train.iloc[chunk:chunk + size, :]

y_train_chunk = y_train.iloc[chunk:chunk + size, :]

random_sample = random.sample(range(0, y_test.shape[0]), np.floor(y_test.shape[0]/10).astype(int))

x_sample_trades = X_train_chunk.iloc[random_sample,:]
y_sample_trades = y_train_chunk.iloc[random_sample, :]

x_sample_history = X_train_chunk.iloc[~np.isin(range(0, y_test.shape[0]), random_sample),:]
x_sample_history = X_train_chunk.iloc[~np.isin(range(0, y_test.shape[0]), random_sample), :]
y_train_sample_history = y_train.iloc[~np.isin(range(0, y_test.shape[0]), random_sample),:]

predictions =\
    xgb_model.predict(xgb.DMatrix(X_train_xgb_chunk.loc[:,X_train_xgb_chunk.columns!='datetime'], label=y_train_chunk))
predictions_sample_trades = predictions[random_sample]
y_train_sample_trades = y_train.iloc[random_sample,:]

nsample_trades=x_sample_trades.shape[0]
nsample_trade_history=x_sample_history.shape[0]

x_sample_trades_knn = x_sample_trades
x_sample_history_knn = x_sample_history

distances = scipy.spatial.distance.cdist(x_sample_trades_knn,
                                         x_sample_history_knn,
                                         metric='euclidean')
distances_spread = scipy.spatial.distance.cdist(y_sample_trades,
                                                y_train_sample_history,
                                                metric='euclidean')

distances_order = distances.argsort(axis=1)


SEED = 4449
random.seed(SEED)
r = 2.0
#for r in [2.0]:
count_missing = 0
range_spread_list = np.empty((0, distances.shape[1]), float)
number_of_neighbors = []
for sample_test in range(0, distances.shape[0]):
    distances_test_sample = distances[sample_test, :]
    range_spread = y_train_sample_history['Spread bps'].values * (distances_test_sample <= r)
    number_of_neighbors.append(sum(range_spread > 0))
    if sum(range_spread > 0) == 0:
        count_missing = count_missing + 1
    else:
        range_spread_list = \
            np.vstack((range_spread_list, range_spread))
df_range_spread = pd.DataFrame(range_spread_list)
samp = random.sample(range(0, df_range_spread.shape[0]), 100)
mask_values = (df_range_spread == 0)

df_range_spread_50p =\
    df_range_spread.mask(mask_values).apply(lambda x: np.percentile(x[~np.isnan(x)], 50), axis=1)
df_range_spread_40p =\
    df_range_spread.mask(mask_values).apply(lambda x: np.percentile(x[~np.isnan(x)], 40), axis=1)
df_range_spread_60p =\
    df_range_spread.mask(mask_values).apply(lambda x: np.percentile(x[~np.isnan(x)], 60), axis=1)
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#df_range_spread.iloc[samp, :].mask(mask_values).T.boxplot(showfliers=False, ax=ax1)
#ax1.plot(range(1, len(predictions_sample_trades[samp]) + 1), predictions_sample_trades[samp], 'r*', label = "XGBooset prediction")
#ax1.plot(range(1, len(predictions_sample_trades[samp]) + 1), y_train_sample_trades['Spread bps'].iloc[samp], 'go', label = "Real spread")
#plt.xticks(rotation=90)
#plt.xlabel('cluster')
#ax1.set_ylabel('spread')
#ax1.legend()
#ax1.set_title('r: ' + str(r) + " | median neigh: " + str(
#    np.median(number_of_neighbors)) + " | perc of total: " + str(
#    np.format_float_positional(100 * np.median(number_of_neighbors) / distances.shape[1], 2)) + "%")
#error_bar = df_range_spread_50p[samp]-#predictions_sample_trades[samp]
plt.figure()
error_bar_up= df_range_spread_60p[samp]-df_range_spread_50p[samp]#error_bar[:]
error_bar_down = df_range_spread_50p[samp]-df_range_spread_40p[samp]#-error_bar[:]
error_bar_up[error_bar_up <0 ] = 0
error_bar_down[error_bar_down <0 ] = 0
plt.errorbar(x=range(1, len(predictions_sample_trades[samp]) +1 ),
             y=df_range_spread_50p[samp],#predictions_sample_trades[samp],
             yerr=[error_bar_down,
                   error_bar_up],
             fmt='o',
             color='black', label = "Percentiles: 40, 50 and 60")
plt.plot(range(1, len(predictions_sample_trades[samp]) + 1), y_train_sample_trades['Spread bps'].iloc[samp], 'go', label = "Real spread")
plt.plot(range(1, len(predictions_sample_trades[samp]) + 1 ), predictions_sample_trades[samp], 'r*', label = "XGBoost prediction")
#plt.set_ylabel('spread')
#plt.set_title('r: ' + str(r) + " | range towards cluster median")
plt.ylabel('spread')
plt.title('r: ' + str(r) + " | range towards cluster median")

plt.legend()

plt.grid()
plt.figure()

df_range_spread_30p =\
    df_range_spread.mask(mask_values).apply(lambda x: np.percentile(x[~np.isnan(x)], 30), axis=1)
df_range_spread_70p =\
    df_range_spread.mask(mask_values).apply(lambda x: np.percentile(x[~np.isnan(x)], 70), axis=1)
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#df_range_spread.iloc[samp, :].mask(mask_values).T.boxplot(showfliers=False, ax=ax1)
#ax1.plot(range(1, len(predictions_sample_trades[samp]) + 1), predictions_sample_trades[samp], 'r*', label = "XGBooset prediction")
#ax1.plot(range(1, len(predictions_sample_trades[samp]) + 1), y_train_sample_trades['Spread bps'].iloc[samp], 'go', label = "Real spread")
#plt.xticks(rotation=90)
#plt.xlabel('cluster')
#ax1.set_ylabel('spread')
#ax1.legend()
#ax1.set_title('r: ' + str(r) + " | median neigh: " + str(
#    np.median(number_of_neighbors)) + " | perc of total: " + str(
#    np.format_float_positional(100 * np.median(number_of_neighbors) / distances.shape[1], 2)) + "%")
error_bar_up= df_range_spread_70p[samp]-df_range_spread_50p[samp]#error_bar[:]
error_bar_down = df_range_spread_50p[samp]-df_range_spread_30p[samp]#-error_bar[:]
error_bar_up[error_bar_up <0 ] = 0
error_bar_down[error_bar_down <0 ] = 0
plt.errorbar(x=range(1, len(predictions_sample_trades[samp]) +1 ),
             y=df_range_spread_50p[samp],#predictions_sample_trades[samp],
             yerr=[error_bar_down,
                   error_bar_up],
             fmt='o',
             color='black', label = "Percentiles: 30, 50 and 70")
plt.plot(range(1, len(predictions_sample_trades[samp]) + 1), y_train_sample_trades['Spread bps'].iloc[samp], 'go', label = "Real spread")
plt.plot(range(1, len(predictions_sample_trades[samp]) + 1 ), predictions_sample_trades[samp], 'r*', label = "XGBoost prediction")
# plt.set_ylabel('spread')
# plt.set_title('r: ' + str(r) + " | range towards cluster median")
plt.ylabel('spread')
plt.title('r: ' + str(r) + " | range towards cluster median")
plt.legend()
plt.grid()

length_range =  df_range_spread_70p[samp]-df_range_spread_50p[samp]
plt.figure()
pd.DataFrame(length_range).boxplot()
plt.figure()
plt.plot(range(length_range.shape[0]), length_range, 'o')
######
######
######
#####
# np.sum(df_range_spread.loc[list(compress(samp, length_range.values>60)),:]>0, axis=1)
# np.sum(df_range_spread.loc[list(compress(samp, length_range.values<15)),:]>0, axis=1)
#
# neighbors = df_range_spread.loc[list(compress(samp, length_range.values>60)),:]>0
# plt.plot(y_train_sample_history.loc[neighbors.iloc[0,:].values, 'Spread bps'], 'o', color='red')
# plt.plot(y_train_sample_history.loc[neighbors.iloc[1,:].values, 'Spread bps'], 'o', color='green')
# print(sum(neighbors.iloc[0,:]))
# print(sum(neighbors.iloc[1,:]))
#
# neighbors = df_range_spread.loc[list(compress(samp, length_range.values>15)),:]>0
# plt.plot(y_train_sample_history.loc[neighbors.iloc[0,:].values, 'Spread bps'], 'o', color='red')
# plt.plot(y_train_sample_history.loc[neighbors.iloc[1,:].values, 'Spread bps'], 'o', color='green')
# print(sum(neighbors.iloc[0,:]))
# print(sum(neighbors.iloc[1,:]))
#
#
# y_train_sample_history.loc[neighbors.iloc[0,:].values, 'Spread bps']
# y_train_sample_history.loc[neighbors.iloc[1,:], 'Spread bps']
#
#f.savefig(root_folder + model_validation_folder + "boxplot-and-range-" + str(n_components_selected) +"-components-rNN-raduis-" + str(r) + ".png")


######
######

X_train_scaled = X_train_scaled[:,[x[0] for x in enumerate(col_correlation_spread) if x[1] > 0.0]]
X_train_scaled_pca = pca.fit_transform(X_train_scaled)
df_train_scaled_pca = pd.DataFrame(data=X_train_scaled_pca)
###

chunk = 0
X_train_chunk = df_train_scaled_pca.iloc[chunk:chunk + size, :]
X_train_xgb_chunk = X_train.iloc[chunk:chunk + size, :]

y_train_chunk = y_train.iloc[chunk:chunk + size, :]

random_sample = random.sample(range(0, y_test.shape[0]), np.floor(y_test.shape[0]/10).astype(int))

x_sample_trades = X_train_chunk.iloc[random_sample,:]
y_sample_trades = y_train_chunk.iloc[random_sample, :]

x_sample_history = X_train_chunk.iloc[~np.isin(range(0, y_test.shape[0]), random_sample),:]
x_sample_history = X_train_chunk.iloc[~np.isin(range(0, y_test.shape[0]), random_sample), :]
y_train_sample_history = y_train.iloc[~np.isin(range(0, y_test.shape[0]), random_sample),:]

predictions =\
    xgb_model.predict(xgb.DMatrix(X_train_xgb_chunk.loc[:,X_train_xgb_chunk.columns!='datetime'], label=y_train_chunk))
predictions_sample_trades = predictions[random_sample]
y_train_sample_trades = y_train.iloc[random_sample,:]

nsample_trades=x_sample_trades.shape[0]
nsample_trade_history=x_sample_history.shape[0]

x_sample_trades_knn = x_sample_trades
x_sample_history_knn = x_sample_history

distances = scipy.spatial.distance.cdist(x_sample_trades_knn,
                                         x_sample_history_knn,
                                         metric='euclidean')
distances_spread = scipy.spatial.distance.cdist(y_sample_trades,
                                                y_train_sample_history,
                                                metric='euclidean')

distances_order = distances.argsort(axis=1)


SEED = 4449
random.seed(SEED)
r = 2.0
#for r in [2.0]:
count_missing = 0
range_spread_list = np.empty((0, distances.shape[1]), float)
number_of_neighbors = []
for sample_test in range(0, distances.shape[0]):
    distances_test_sample = distances[sample_test, :]
    range_spread = y_train_sample_history['Spread bps'].values * (distances_test_sample <= r)
    number_of_neighbors.append(sum(range_spread > 0))
    if sum(range_spread > 0) == 0:
        count_missing = count_missing + 1
    else:
        range_spread_list = \
            np.vstack((range_spread_list, range_spread))
df_range_spread = pd.DataFrame(range_spread_list)
samp = random.sample(range(0, df_range_spread.shape[0]), 100)
mask_values = (df_range_spread == 0)

df_range_spread_50p =\
    df_range_spread.mask(mask_values).apply(lambda x: np.percentile(x[~np.isnan(x)], 50), axis=1)
df_range_spread_40p =\
    df_range_spread.mask(mask_values).apply(lambda x: np.percentile(x[~np.isnan(x)], 40), axis=1)
df_range_spread_60p =\
    df_range_spread.mask(mask_values).apply(lambda x: np.percentile(x[~np.isnan(x)], 60), axis=1)
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#df_range_spread.iloc[samp, :].mask(mask_values).T.boxplot(showfliers=False, ax=ax1)
#ax1.plot(range(1, len(predictions_sample_trades[samp]) + 1), predictions_sample_trades[samp], 'r*', label = "XGBooset prediction")
#ax1.plot(range(1, len(predictions_sample_trades[samp]) + 1), y_train_sample_trades['Spread bps'].iloc[samp], 'go', label = "Real spread")
#plt.xticks(rotation=90)
#plt.xlabel('cluster')
#ax1.set_ylabel('spread')
#ax1.legend()
#ax1.set_title('r: ' + str(r) + " | median neigh: " + str(
#    np.median(number_of_neighbors)) + " | perc of total: " + str(
#    np.format_float_positional(100 * np.median(number_of_neighbors) / distances.shape[1], 2)) + "%")
#error_bar = df_range_spread_50p[samp]-#predictions_sample_trades[samp]
plt.figure()
error_bar_up= df_range_spread_60p[samp]-df_range_spread_50p[samp]#error_bar[:]
error_bar_down = df_range_spread_50p[samp]-df_range_spread_40p[samp]#-error_bar[:]
error_bar_up[error_bar_up <0 ] = 0
error_bar_down[error_bar_down <0 ] = 0
plt.errorbar(x=range(1, len(predictions_sample_trades[samp]) +1 ),
             y=df_range_spread_50p[samp],#predictions_sample_trades[samp],
             yerr=[error_bar_down,
                   error_bar_up],
             fmt='o',
             color='black', label = "Percentiles: 40, 50 and 60")
plt.plot(range(1, len(predictions_sample_trades[samp]) + 1), y_train_sample_trades['Spread bps'].iloc[samp], 'go', label = "Real spread")
plt.plot(range(1, len(predictions_sample_trades[samp]) + 1 ), predictions_sample_trades[samp], 'r*', label = "XGBoost prediction")
#plt.set_ylabel('spread')
#plt.set_title('r: ' + str(r) + " | range towards cluster median")
plt.ylabel('spread')
plt.title('r: ' + str(r) + " | range towards cluster median")

plt.legend()

plt.grid()
plt.figure()

df_range_spread_30p =\
    df_range_spread.mask(mask_values).apply(lambda x: np.percentile(x[~np.isnan(x)], 30), axis=1)
df_range_spread_70p =\
    df_range_spread.mask(mask_values).apply(lambda x: np.percentile(x[~np.isnan(x)], 70), axis=1)
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#df_range_spread.iloc[samp, :].mask(mask_values).T.boxplot(showfliers=False, ax=ax1)
#ax1.plot(range(1, len(predictions_sample_trades[samp]) + 1), predictions_sample_trades[samp], 'r*', label = "XGBooset prediction")
#ax1.plot(range(1, len(predictions_sample_trades[samp]) + 1), y_train_sample_trades['Spread bps'].iloc[samp], 'go', label = "Real spread")
#plt.xticks(rotation=90)
#plt.xlabel('cluster')
#ax1.set_ylabel('spread')
#ax1.legend()
#ax1.set_title('r: ' + str(r) + " | median neigh: " + str(
#    np.median(number_of_neighbors)) + " | perc of total: " + str(
#    np.format_float_positional(100 * np.median(number_of_neighbors) / distances.shape[1], 2)) + "%")
error_bar_up= df_range_spread_70p[samp]-df_range_spread_50p[samp]#error_bar[:]
error_bar_down = df_range_spread_50p[samp]-df_range_spread_30p[samp]#-error_bar[:]
error_bar_up[error_bar_up <0 ] = 0
error_bar_down[error_bar_down <0 ] = 0
plt.errorbar(x=range(1, len(predictions_sample_trades[samp]) +1 ),
             y=df_range_spread_50p[samp],#predictions_sample_trades[samp],
             yerr=[error_bar_down,
                   error_bar_up],
             fmt='o',
             color='black', label = "Percentiles: 30, 50 and 70")
plt.plot(range(1, len(predictions_sample_trades[samp]) + 1), y_train_sample_trades['Spread bps'].iloc[samp], 'go', label = "Real spread")
plt.plot(range(1, len(predictions_sample_trades[samp]) + 1 ), predictions_sample_trades[samp], 'r*', label = "XGBoost prediction")
# plt.set_ylabel('spread')
# plt.set_title('r: ' + str(r) + " | range towards cluster median")
plt.ylabel('spread')
plt.title('r: ' + str(r) + " | range towards cluster median")
plt.legend()
plt.grid()

length_range =  df_range_spread_70p[samp]-df_range_spread_50p[samp]
plt.figure()
pd.DataFrame(length_range).boxplot()
plt.figure()
plt.plot(range(length_range.shape[0]), length_range, 'o')
######
######
######
#####
# np.sum(df_range_spread.loc[list(compress(samp, length_range.values>60)),:]>0, axis=1)
# np.sum(df_range_spread.loc[list(compress(samp, length_range.values<15)),:]>0, axis=1)
#
# neighbors = df_range_spread.loc[list(compress(samp, length_range.values>60)),:]>0
# plt.plot(y_train_sample_history.loc[neighbors.iloc[0,:].values, 'Spread bps'], 'o', color='red')
# plt.plot(y_train_sample_history.loc[neighbors.iloc[1,:].values, 'Spread bps'], 'o', color='green')
# print(sum(neighbors.iloc[0,:]))
# print(sum(neighbors.iloc[1,:]))
#
# neighbors = df_range_spread.loc[list(compress(samp, length_range.values>15)),:]>0
# plt.plot(y_train_sample_history.loc[neighbors.iloc[0,:].values, 'Spread bps'], 'o', color='red')
# plt.plot(y_train_sample_history.loc[neighbors.iloc[1,:].values, 'Spread bps'], 'o', color='green')
# print(sum(neighbors.iloc[0,:]))
# print(sum(neighbors.iloc[1,:]))
#
#
# y_train_sample_history.loc[neighbors.iloc[0,:].values, 'Spread bps']
# y_train_sample_history.loc[neighbors.iloc[1,:], 'Spread bps']
#
#f.savefig(root_folder + model_validation_folder + "boxplot-and-range-" + str(n_components_selected) +"-components-rNN-raduis-" + str(r) + ".png")

