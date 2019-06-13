from IPython import get_ipython;
get_ipython().magic('reset -sf')

import re
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pickle
from sklearn.decomposition import PCA
import scipy

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
with open(root_folder + model_object_folder + xgboost_mse_bias_output_file, "rb") as pickle_mse_bias_sq_output_file:
    mse_bias_sq_xgboost_estimator = pickle.load(pickle_mse_bias_sq_output_file)
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

df_measures_knn = \
    pd.DataFrame(columns=['n_components',
                          'algorithm',
                          'kr',
                          'count_missing_neighbors',
                          'chunk',
                          'nsample_trades',
                          'nsample_trade_history',
                          'mse_mean_knn',
                          'variance_mean_knn',
                          'bias_squared_mean_knn',
                          'bias_abs_squared_mean_knn',
                          'mse_median_knn',
                          'variance_median_knn',
                          'bias_squared_median_knn',
                          'bias_abs_squared_median_knn',
                          'mse_xgboost',
                          'variance_xgboost',
                          'bias_squared_xgboost',
                          'bias_abs_squared_xgboost',
                          'coverage_10_90',
                          'coverage_25_75',
                          'coverage_40_60',
                          'distance_10_90_to_20_bps_zeros',
                          'distance_10_90_to_20_bps_average',
                          'distance_10_90_to_20_bps_q5',
                          'distance_10_90_to_20_bps_q10',
                          'distance_10_90_to_20_bps_q90',
                          'distance_10_90_to_20_bps_q95',
                          'distance_25_75_to_20_bps_zeros',
                          'distance_25_75_to_20_bps_average',
                          'distance_25_75_to_20_bps_q5',
                          'distance_25_75_to_20_bps_q10',
                          'distance_25_75_to_20_bps_q90',
                          'distance_25_75_to_20_bps_q95',
                          'distance_40_60_to_20_bps_zeros',
                          'distance_40_60_to_20_bps_average',
                          'distance_40_60_to_20_bps_q5',
                          'distance_40_60_to_20_bps_q10',
                          'distance_40_60_to_20_bps_q90',
                          'distance_40_60_to_20_bps_q95',
                          'mean_nneighbors',
                          'median_nneighbors'])

for n_components_selected in [10, 5, 3, 2]:

    scaler = preprocessing.StandardScaler()
    pca = PCA(n_components=n_components_selected)
    reference_spread_bps = 20

    X_train_selected_vars = X_train.loc[:, np.isin(X_train.columns, features_importance['Feature'].unique())]
    X_train_scaled = scaler.fit_transform(X_train_selected_vars)
    X_train_scaled_pca = pca.fit_transform(X_train_scaled)
    df_train_scaled_pca = pd.DataFrame(data=X_train_scaled_pca)

    # for col in range(0, df_train_scaled_pca.shape[1]):
        # plt.figure()
        # plt.hist(df_train_scaled_pca .iloc[:, col], bins=100)
        # plt.title("component - " + str(col))
        # plt.savefig(root_folder + model_validation_folder + "histogram-component-" + str(col) + "-of-" + str(n_components_selected) + "components.png")

    for kNN_or_radius in ['knn', 'radius']:

        if kNN_or_radius == 'knn':
            parameters_list = [120, 100, 80, 60, 40, 20]
        else:
            #parameters_list = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            parameters_list = [30.0,
                               25.0,
                               20.0,
                               15.0,
                               14.0,
                               13.0,
                               12.0,
                               11.0,
                               10.0,
                               9.0,
                               8.0,
                               7.0,
                               6.0,
                               5.0,
                               4.0,
                               3.0,
                               2.0,
                               1.5,
                               1.0]
        chunk_id = 0
        for chunk in range(0, df_train_scaled_pca.shape[0], size):

            X_train_chunk = df_train_scaled_pca.iloc[chunk:chunk + size, :]
            y_train_chunk = y_train.iloc[chunk:chunk + size, :]

            x_sample_history,x_sample_trades, y_sample_history, y_sample_trades =\
                train_test_split(X_train_chunk, y_train_chunk, test_size=0.10, random_state=42)

            nsample_trades=x_sample_trades.shape[0]
            nsample_trade_history=x_sample_history.shape[0]

            x_sample_trades_knn = x_sample_trades
            x_sample_history_knn = x_sample_history

            distances = scipy.spatial.distance.cdist(x_sample_trades_knn,
                                                     x_sample_history_knn,
                                                     metric='euclidean')
            distances_order = distances.argsort(axis=1)

            for kr in parameters_list:
                print("components: " + str(n_components_selected) + " | algorithm : " + kNN_or_radius + " | chunk: " + str(chunk_id) + " | kr: " + str(kr))
                count_missing = 0
                number_of_neighbors = []
                if kNN_or_radius == 'knn':
                    range_spread_list = np.empty((0, kr), float)
                    distances_sample_trade = pd.DataFrame(distances_order[:, 0:kr])
                    real_spreads = y_sample_trades['Spread bps'].values
                    for sample_trade in range(0, distances_order.shape[0]):
                        range_spread = y_sample_history['Spread bps'].iloc[distances_sample_trade.iloc[sample_trade, :].values]
                        range_spread_list = \
                            np.vstack((range_spread_list, range_spread))
                    ############
                    df_range_spread = pd.DataFrame(range_spread_list)
                    ############
                    df_range_spread_90p = df_range_spread.apply(lambda x: np.percentile(x, 90), axis=1)
                    df_range_spread_75p = df_range_spread.apply(lambda x: np.percentile(x, 75), axis=1)
                    df_range_spread_60p = df_range_spread.apply(lambda x: np.percentile(x, 60), axis=1)
                    df_range_spread_50p = df_range_spread.apply(lambda x: np.percentile(x, 50), axis=1)
                    df_range_spread_40p = df_range_spread.apply(lambda x: np.percentile(x, 40), axis=1)
                    df_range_spread_25p = df_range_spread.apply(lambda x: np.percentile(x, 25), axis=1)
                    df_range_spread_10p = df_range_spread.apply(lambda x: np.percentile(x, 25), axis=1)
                    mean_knn = df_range_spread.apply(lambda x: np.mean(x), axis=1)
                    median_knn = df_range_spread_50p
                    number_of_neighbors.append(kr)
                else:
                    df_range_spread_90p = []
                    df_range_spread_75p = []
                    df_range_spread_60p = []
                    df_range_spread_50p = []
                    df_range_spread_40p = []
                    df_range_spread_25p = []
                    df_range_spread_10p = []
                    mean_knn = []
                    real_spreads = []
                    for sample_trade in range(0, distances_order.shape[0]):
                        distances_sample_trade = distances[sample_trade, :]
                        range_spread = y_sample_history['Spread bps'].iloc[(distances_sample_trade<=kr)]
                        number_of_neighbors.append(len(range_spread))
                        if len(range_spread)==0:
                            count_missing = count_missing + 1
                        else:
                            df_range_spread_90p.append(np.percentile(range_spread, 90))
                            df_range_spread_75p.append(np.percentile(range_spread, 75))
                            df_range_spread_60p.append(np.percentile(range_spread, 60))
                            df_range_spread_50p.append(np.percentile(range_spread, 50))
                            df_range_spread_40p.append(np.percentile(range_spread, 40))
                            df_range_spread_25p.append(np.percentile(range_spread, 25))
                            df_range_spread_10p.append(np.percentile(range_spread, 10))
                            real_spreads.append(y_sample_trades['Spread bps'].iloc[sample_trade])
                            mean_knn.append(np.mean(range_spread))
                    df_range_spread_90p = pd.Series(df_range_spread_90p)
                    df_range_spread_75p = pd.Series(df_range_spread_75p)
                    df_range_spread_60p = pd.Series(df_range_spread_60p)
                    df_range_spread_40p = pd.Series(df_range_spread_40p)
                    df_range_spread_25p = pd.Series(df_range_spread_25p)
                    df_range_spread_10p = pd.Series(df_range_spread_10p)
                    mean_knn = pd.Series(mean_knn)

                median_knn = df_range_spread_50p
                mse_mean_estim = mean_squared_error(real_spreads, mean_knn)
                var_mean_estim = np.var(mean_knn)
                bias_sq_mean_estim = (np.average(np.array(real_spreads) - np.array(mean_knn)))**2
                bias_abs_sq_mean_estim = (np.average(np.abs(np.array(real_spreads) - np.array(mean_knn))))**2

                mse_median_estim = mean_squared_error(real_spreads, median_knn)
                var_median_estim = np.var(median_knn)
                bias_sq_median_estim = (np.average(np.array(real_spreads) - np.array(median_knn)))**2
                bias_abs_sq_median_estim = (np.average(np.abs(np.array(real_spreads) - np.array(median_knn))))**2

                coverage_10_90 = \
                    sum((df_range_spread_10p <= real_spreads) &
                        (df_range_spread_90p >= real_spreads)) /y_sample_trades.shape[0]
                coverage_40_60 = \
                    sum((df_range_spread_40p <= real_spreads) &
                        (df_range_spread_60p >= real_spreads)) /y_sample_trades.shape[0]
                coverage_25_75 = \
                    sum((df_range_spread_25p <= real_spreads) &
                        (df_range_spread_75p >= real_spreads)) /y_sample_trades.shape[0]
                # distance range 10p-90p to 20bps
                distance_10_90_to_20_bps = (df_range_spread_90p - df_range_spread_10p)-20
                distance_10_90_to_20_bps[distance_10_90_to_20_bps < 0] = 0
                distance_10_90_to_20_bps_zeros = sum(distance_10_90_to_20_bps==0)
                distance_10_90_to_20_bps_average = np.mean(distance_10_90_to_20_bps)
                distance_10_90_to_20_bps_q5 = np.percentile(distance_10_90_to_20_bps, 5)
                distance_10_90_to_20_bps_q10 = np.percentile(distance_10_90_to_20_bps,10)
                distance_10_90_to_20_bps_q90 = np.percentile(distance_10_90_to_20_bps,90)
                distance_10_90_to_20_bps_q95 = np.percentile(distance_10_90_to_20_bps,95)
                # distance range 25p-75p to 20bps
                distance_25_75_to_20_bps = (df_range_spread_75p - df_range_spread_25p)-20
                distance_25_75_to_20_bps[distance_25_75_to_20_bps < 0] = 0
                distance_25_75_to_20_bps_zeros = sum(distance_25_75_to_20_bps==0)
                distance_25_75_to_20_bps_average = np.mean(distance_25_75_to_20_bps)
                distance_25_75_to_20_bps_q5 = np.percentile(distance_25_75_to_20_bps, 5)
                distance_25_75_to_20_bps_q10 = np.percentile(distance_25_75_to_20_bps, 10)
                distance_25_75_to_20_bps_q90 = np.percentile(distance_25_75_to_20_bps, 90)
                distance_25_75_to_20_bps_q95 = np.percentile(distance_25_75_to_20_bps, 95)
                # distance range 40p-60p to 20bps
                distance_40_60_to_20_bps = (df_range_spread_60p - df_range_spread_40p)-20
                distance_40_60_to_20_bps[distance_40_60_to_20_bps < 0] = 0
                distance_40_60_to_20_bps_zeros = sum(distance_40_60_to_20_bps==0)
                distance_40_60_to_20_bps_average = np.mean(distance_40_60_to_20_bps)
                distance_40_60_to_20_bps_q5 = np.percentile(distance_40_60_to_20_bps, 5)
                distance_40_60_to_20_bps_q10 = np.percentile(distance_40_60_to_20_bps, 10)
                distance_40_60_to_20_bps_q90 = np.percentile(distance_40_60_to_20_bps, 90)
                distance_40_60_to_20_bps_q95 = np.percentile(distance_40_60_to_20_bps, 95)

                mean_nneighbors = np.average(number_of_neighbors)
                median_nneighbors = np.median(number_of_neighbors)

                df_measures_knn =\
                    df_measures_knn.append({
                                           'n_components': n_components_selected,
                                           'algorithm':kNN_or_radius,
                                           'kr':kr,
                                           'count_missing_neighbors':count_missing,
                                           'chunk':chunk_id,
                                           'nsample_trades':nsample_trades,
                                           'nsample_trade_history' : nsample_trade_history,
                                           'mse_mean_knn':mse_mean_estim,
                                           'variance_mean_knn':var_mean_estim,
                                           'bias_squared_mean_knn':bias_sq_mean_estim,
                                           'bias_abs_squared_mean_knn':bias_abs_sq_mean_estim,
                                           'mse_median_knn':mse_median_estim,
                                           'variance_median_knn':var_median_estim,
                                           'bias_squared_median_knn':bias_sq_median_estim,
                                           'bias_abs_squared_median_knn':bias_abs_sq_median_estim,
                                           'mse_xgboost':mse_total_xgboost_estimator,
                                           'variance_xgboost':mse_var_xgboost_estimator,
                                           'bias_squared_xgboost':mse_bias_sq_xgboost_estimator,
                                           'bias_abs_squared_xgboost':mse_bias_abs_sq_xgboost_estimator,
                                           'coverage_10_90':coverage_10_90,
                                           'coverage_25_75':coverage_25_75,
                                           'coverage_40_60':coverage_40_60,
                                           'distance_10_90_to_20_bps_zeros':distance_10_90_to_20_bps_zeros,
                                           'distance_10_90_to_20_bps_average':distance_10_90_to_20_bps_average,
                                           'distance_10_90_to_20_bps_q5':distance_10_90_to_20_bps_q5,
                                           'distance_10_90_to_20_bps_q10':distance_10_90_to_20_bps_q10,
                                           'distance_10_90_to_20_bps_q90':distance_10_90_to_20_bps_q90,
                                           'distance_10_90_to_20_bps_q95':distance_10_90_to_20_bps_q95,
                                           'distance_25_75_to_20_bps_zeros':distance_25_75_to_20_bps_zeros,
                                           'distance_25_75_to_20_bps_average':distance_25_75_to_20_bps_average,
                                           'distance_25_75_to_20_bps_q5':distance_25_75_to_20_bps_q5,
                                           'distance_25_75_to_20_bps_q10':distance_25_75_to_20_bps_q10,
                                           'distance_25_75_to_20_bps_q90':distance_25_75_to_20_bps_q90,
                                           'distance_25_75_to_20_bps_q95':distance_25_75_to_20_bps_q95,
                                           'distance_40_60_to_20_bps_zeros':distance_40_60_to_20_bps_zeros,
                                           'distance_40_60_to_20_bps_average':distance_40_60_to_20_bps_average,
                                           'distance_40_60_to_20_bps_q5':distance_40_60_to_20_bps_q5,
                                           'distance_40_60_to_20_bps_q10':distance_40_60_to_20_bps_q10,
                                           'distance_40_60_to_20_bps_q90':distance_40_60_to_20_bps_q90,
                                           'distance_40_60_to_20_bps_q95':distance_40_60_to_20_bps_q95,
                                           'mean_nneighbors':mean_nneighbors,
                                           'median_nneighbors':median_nneighbors},
                        ignore_index = True)
            chunk_id = chunk_id+1
df_measures_knn.to_excel(root_folder + model_validation_folder + "df_measures_knn-chunks.xlsx", index=False)

