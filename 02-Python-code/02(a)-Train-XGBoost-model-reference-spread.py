from IPython import get_ipython;
get_ipython().magic('reset -sf')

import re
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import xgboost as xgb
import pickle
import bisect
import sklearn.model_selection.GridSearchCV

# folders names definitions
root_folder = 'C:/Users/Pablo Andrade/Documents/2019-01-Jan/01-BTU-FX-Pricing/09-Hand-over/'
root_folder = 'C:/Users/Pablo Andrade/Box Sync/FX Pricing Model/09-Handover/'
data_folder = '01-Data/'
raw_data_folder = '01-Raw datasets/'
final_dataset_folder = '02-Final dataset/'
model_visualization_folder = '04-Models visualization/'
implementation_dataset_folder = "03-Implementation/"
model_validation_folder = '05-Model validation/'
model_object_folder = '06-Models objects (pickle)/'
implementation_dataset_folder = "03-Implementation/"

# files names definitions
final_dataset_file_csv = '01-FXTrades_ClientInfo_ShareOfWallet_MarketData.csv'
final_dataset_file_csv = "01-FXTrades_ClientInfo_ShareOfWallet_MarketData_newdata.csv"
features_importance_rf_csv = 'varImp-randomforest.csv'
features_importance_xgboost_csv = 'varImp-XGBoost.csv'
features_gain_xgboost_csv = 'varImp-gain-XGBoost.csv'
features_correlation_with_target_csv = 'features_correlatoin-with-target.csv'
xgboost_reference_price_model = 'xgboost_reference_price_model.pickle'
features_xgboost_file = 'features_xgboost_file.pickle'
features_xgboost_gain_file = 'features_xgboost_gain_file.pickle'
encoders_pickle_file = 'encoders_pickle_file.pickle'
X_train_file = 'X_train.pickle'
X_test_file = 'X_test.pickle'
y_train_file = 'y_train.pickle'
y_test_file = 'y_test.pickle'
input_data_for_kNN = 'input_data_for_kNN.pickle'

xgboost_mse_total_output_file = 'xgboost_mse_total_output_file.pickle'
xgboost_mse_var_output_file = 'xgboost_mse_var_output_file.pickle'
xgboost_mse_bias_output_file = 'xgboost_mse_bias_output_file.pickle'
xgboost_mse_bias_abs_output_file = 'xgboost_mse_bias_abs_output_file.pickle'

negative_spreads_file = "Check-Trades-Negative_spreads.xlsx"
over300_spreads_file = "Check-Trades-Spreads_over_300.xlsx"
all_training_features_file = "all_training_features.xlsx"
numeric_training_features_file = "numeric_training_features.xlsx"
categorical_training_features_file = "categorical_training_features.xlsx"
default_values_for_missing_file = "default_values_for_missing.xlsx"

plot_rf_predctionVSreal_file = 'RF_predction_vs_real.png'
plot_training_validation_file = 'XGBoost_training_vs_validation_error.png'
html_xgb_file1 = 'html_file1.html'

# read dataset file
input_data = pd.read_csv(root_folder + data_folder + final_dataset_folder + final_dataset_file_csv,
                         sep =';',
                         decimal =',')

# create target variable (new variable of spread in base points)
# variable target
target_variable = ['Spread bps']
input_data['Spread bps'] = 10000 * (input_data['Margem USD'] / input_data['Volume USD'])

input_data_neg_spreads = input_data.loc[input_data['Spread bps'] <= 0]

# writer = pd.ExcelWriter(root_folder + data_folder + raw_data_folder + negative_spreads_file)
# input_data_neg_spreads.to_excel(writer,"data", index =False)
# writer.save()

input_data = input_data.loc[input_data['Spread bps'] > 0]
input_data_higher_than_300_spreads = input_data.loc[input_data['Spread bps'] > 300]

# writer = pd.ExcelWriter(root_folder + data_folder + raw_data_folder + over300_spreads_file)
# input_data_higher_than_300_spreads.to_excel(writer,"data", index =False)
# writer.save()

input_data = input_data.loc[input_data['Spread bps'] < 300]

targets = input_data[target_variable]
datetime = input_data['datetime']
client_IDs = input_data['Cod Grupo McK']
volume = input_data['Volume USD']
margin = input_data['Margem USD']
# ID variables (variable for identification of user/date)
id_variables = ['Cod Grupo McK', 'datetime']

trades_per_client = input_data[['Cod Grupo McK', 'datetime', 'Volume USD', 'Spread bps']]
trades_per_client.sort_values(by=['Cod Grupo McK', 'datetime'], inplace=True)
trades_per_client = trades_per_client.groupby(['Cod Grupo McK']).tail(3).reset_index()
trades_per_client = trades_per_client[['Cod Grupo McK', 'datetime', 'Volume USD', 'Spread bps']]

# variables duplicated (other variables have same meaning)
duplicated_variables = ['Margem USD','Rating_Grupo']
# variables to be removed (not necessary to model)
remove_variables = ['CAM57_Região', 'CAM57_Plat','Tipo Contabil','Regiao','Transacoes_Anteriores_Dia']
# variables with too many factors
variables_too_many_factors = ['Pais Origem Capital','Setor Economico', 'Setor 38','Setor 30 Raw']
# numierc variables
numeric_variables = ['Volume USD','Taxa Base','PRAZO_REAL', 'PRAZO_USD','SoC','Share SV Bradesco','Share SV BB','Share SV Santander','Share SV City','Share SV Caixa','Share SV Safra',
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
                     'MX_PESO_VOLAT_6_Hours', 'MX_PESO_VOLAT_4_Hours','MX_PESO_VOLAT_2_Hours',
                     'QTD CONTRATOS', 'CASADO_Curncy_Bid_minus_Ask_Price','CASADO_Curncy_Bid_Price',
                     'CASADO_Curncy_Ask_Price', 'HORA_LOCAL', 'Corretora', 'Produto']
# categorical variables
categorical_variables = ['Tipo Capital','Auditado Por','Fonte do Balanco','Auditoria Balanco','Setor 30',
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
###
default_values_for_missing =\
    pd.DataFrame(columns=['column_name', 'value'])
#####
# numeric variables
# #####
# PB_ratio and PB_ratio_ex_IB: string to numeric variables
input_data.loc[input_data['logPD'].isna(),'logPD'] = \
    input_data.loc[~input_data['logPD'].isna(),'logPD'].min()
default_values_for_missing =\
    default_values_for_missing.append(
        {'column_name': 'logPD',
         'value': input_data.loc[~input_data['logPD'].isna(),'logPD'].min()},
        ignore_index=True)

for variable in numeric_variables:
    input_data.loc[input_data[variable].isna(), variable] =\
        input_data.loc[~input_data[variable].isna(),variable].mean()
    default_values_for_missing = \
        default_values_for_missing.append(
            {'column_name': variable,
             'value': input_data.loc[~input_data[variable].isna(),variable].mean()},
            ignore_index=True)

for variable in numeric_variables:
    input_data[variable] = input_data[variable].astype(float)
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
    default_values_for_missing = \
        default_values_for_missing.append(
            {'column_name': variable,
             'value': 2},
            ignore_index=True)

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
# transform categorical variable using Label encoder
list_label_enc = []
for column in categorical_variables:
    le = preprocessing.LabelEncoder()
    le.fit(input_data[column].astype(str))
    le_classes = le.classes_.tolist()
    bisect.insort_left(le_classes, '<unknown>')
    le.classes_ = le_classes
    list_label_enc.append(le)
    input_data[column] = le.transform(input_data[column].astype(str))
    default_values_for_missing = \
        default_values_for_missing.append(
            {'column_name': variable,
             'value': le.classes_.index('<unknown>')},
            ignore_index=True)
print(default_values_for_missing)

# save encoders list to disk
with open(root_folder + model_object_folder + encoders_pickle_file, "wb") as pickle_encoders_file:
    pickle.dump(list_label_enc, pickle_encoders_file)

# correlation
correlations = {}
#features
for f in numeric_variables:
    x1 = input_data[f].values
    x2 = targets['Spread bps'].values
    #key = f + ' vs ' + targets.columns[0]
    key = f
    correlations[key] = stats.pearsonr(x1, x2)[0]
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations = data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]
data_correlations = data_correlations.reset_index()
data_correlations.columns = ['Feature','Correlation with target (spread in bps)']
data_correlations.to_csv(root_folder + model_validation_folder + features_correlation_with_target_csv, sep=';', decimal=',', index=False)

regex = re.compile(r"\[|\]|<", re.IGNORECASE)
input_data.columns =\
    [regex.sub("_", col)
     if any(x in str(col)
            for x in set(('[', ']', '<')))
     else col for col in input_data.columns.values]

input_data_forkNN = input_data.copy(deep =True)

input_data_forkNN['datetime'] = datetime
input_data_forkNN['Cod Grupo McK'] = client_IDs
input_data_forkNN['Volume USD'] = volume
input_data_forkNN['Spread bps'] = targets
input_data_forkNN['Margem USD'] = margin

## save dataset for kNN
with open(root_folder + model_object_folder + input_data_for_kNN, "wb") as pickle_kNN_input_data_file:
    pickle.dump(input_data_forkNN, pickle_kNN_input_data_file)

all_training_features = pd.DataFrame(input_data.columns)
numeric_training_features = pd.DataFrame(input_data.columns[np.isin(input_data.columns, numeric_variables)])
categorical_training_features = pd.DataFrame(input_data.columns[np.isin(input_data.columns, categorical_variables)])

all_training_features.to_excel(root_folder + model_validation_folder + all_training_features_file)
numeric_training_features.to_excel(root_folder + model_validation_folder + numeric_training_features_file)
categorical_training_features.to_excel(root_folder + model_validation_folder + categorical_training_features_file)

#####
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data, targets, test_size=0.25, random_state=42)

# save the training and validation datatsets to disk
with open(root_folder + model_object_folder + X_train_file, "wb") as pickle_X_train_file:
    pickle.dump(X_train, pickle_X_train_file)
with open(root_folder + model_object_folder + X_test_file, "wb") as pickle_X_test_file:
    pickle.dump(X_test, pickle_X_test_file)
with open(root_folder + model_object_folder + y_train_file, "wb") as pickle_y_train_file:
    pickle.dump(y_train, pickle_y_train_file)
with open(root_folder + model_object_folder + y_test_file, "wb") as pickle_y_test_file:
    pickle.dump(y_test, pickle_y_test_file)

train_xgb = xgb.DMatrix(X_train, label=y_train)
test_xgb = xgb.DMatrix(X_test, label=y_test)

params = {'objective': 'reg:linear', 'eval_metric': 'rmse', 'booster': 'gbtree', "max_depth": 6}
xgb_table = xgb.cv(params, train_xgb, 100, nfold=3, early_stopping_rounds=20, verbose_eval=True)
xgb_model = xgb.train(params, train_xgb, max(xgb_table.index))

## grid search cross validation
param_test5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}

gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
 param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

# save the model to disk
with open(root_folder + model_object_folder + xgboost_reference_price_model, "wb") as pickle_output_file:
    pickle.dump(xgb_model, pickle_output_file)

plt.figure()
plt.plot(xgb_table['train-rmse-mean'], label = "training set")
plt.plot(xgb_table['test-rmse-mean'], label = "validation set")
plt.title("RMSE - XGBoost - train vs test")
plt.legend()
plt.savefig(root_folder + model_validation_folder + plot_training_validation_file)

predictions = xgb_model.predict(test_xgb)
print(predictions)

plt.figure()
plt.plot(predictions, y_test['Spread bps'].values, 'o')
plt.plot(np.unique(y_test['Spread bps'].values),
         np.poly1d(np.polyfit(y_test['Spread bps'].values, predictions, 1))(np.unique(y_test['Spread bps'].values)))
plt.plot(np.unique(y_test['Spread bps'].values),
         np.unique(y_test['Spread bps'].values))
plt.xlabel('Predicted spread')
plt.ylabel('Real spread')
plt.title('XGBoost - predicted vs real spread (r2: ' +
          str(round(r2_score(y_test['Spread bps'].values, predictions),2)) + ")")
plt.savefig(root_folder + model_validation_folder + plot_rf_predctionVSreal_file)

mse_total_xgboost_estimator = mean_squared_error(predictions, y_test['Spread bps'].values)
mse_var_xgboost_estimator = np.var(predictions)
mse_bias_sq_xgboost_estimator = (np.average(predictions-y_test['Spread bps'].values))**2
mse_bias_abs_sq_xgboost_estimator = (np.average(abs(predictions-y_test['Spread bps'].values)))**2
# save the model to disk
with open(root_folder + model_object_folder + xgboost_mse_total_output_file, "wb") as pickle_mse_total_output_file:
    pickle.dump(mse_total_xgboost_estimator, pickle_mse_total_output_file)
with open(root_folder + model_object_folder + xgboost_mse_var_output_file, "wb") as pickle_mse_var_output_file:
    pickle.dump(mse_var_xgboost_estimator, pickle_mse_var_output_file)
with open(root_folder + model_object_folder + xgboost_mse_bias_output_file, "wb") as pickle_mse_bias_sq_output_file:
    pickle.dump(mse_bias_sq_xgboost_estimator, pickle_mse_bias_sq_output_file)
with open(root_folder + model_object_folder + xgboost_mse_bias_abs_output_file, "wb") as pickle_mse_bias_abs_sq_output_file:
    pickle.dump(mse_bias_abs_sq_xgboost_estimator, pickle_mse_bias_abs_sq_output_file)
# get_score(fmap='', importance_type='weight')
#
# fmap (str (optional)) – The name of feature map file.
# importance_type
# ‘weight’ - the number of times a feature is used to split the data across all trees.
# ‘gain’ - the average gain across all splits the feature is used in.
# ‘cover’ - the average coverage across all splits the feature is used in.
# ‘total_gain’ - the total gain across all splits the feature is used in.
# ‘total_cover’ - the total coverage across all splits the feature is used in.

features_gain = pd.DataFrame.from_dict(xgb_model.get_score(importance_type='gain'), orient='index')
features_gain = features_gain.reset_index()
features_gain.columns = ['Feature','Gain']
with open(root_folder + model_object_folder + features_xgboost_gain_file, "wb") as pickle_features_gain_file:
    pickle.dump(features_gain, pickle_features_gain_file)
features_gain.to_csv(root_folder + model_validation_folder + features_gain_xgboost_csv, sep=';', decimal=',', index=False)

features = pd.DataFrame.from_dict(xgb_model.get_fscore(), orient='index')
features = features.reset_index()
features.columns = ['Feature','Importance']
with open(root_folder + model_object_folder + features_xgboost_file, "wb") as pickle_features_file:
    pickle.dump(features, pickle_features_file)
features.to_csv(root_folder + model_validation_folder + features_importance_xgboost_csv, sep=';', decimal=',', index=False)
#xgb.plot_importance(xgb_model)

default_values_for_missing.to_excel(
    root_folder + data_folder + implementation_dataset_folder + default_values_for_missing_file)

# Calculate the absolute errors
errors = abs(predictions - y_test.values)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Print out the mean absolute error (mae)
print('Mean Squared Error:', round(mean_squared_error(y_test.values, predictions), 2))

print('Training data Shape:', X_train.shape)
print('Training targets Shape:', y_train.shape)
print('Testing data Shape:', X_test.shape)
print('Testing targets Shape:', y_test.shape)
