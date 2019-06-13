from IPython import get_ipython;
get_ipython().magic('reset -sf')

import pandas as pd
import numpy as np
import pickle
from itertools import compress

# folders names definitions
root_folder = "C:/Users/Pablo Andrade/Box Sync/FX Pricing Model/09-Handover/"
data_folder = "01-Data/"
raw_data_folder = "01-Raw datasets/"
implementation_dataset_folder = "03-Implementation/"
model_validation_folder = '05-Model validation/'
model_object_folder = '06-Models objects (pickle)/'
all_training_features_file = "all_training_features.xlsx"
numeric_training_features_file = "numeric_training_features.xlsx"
categorical_training_features_file = "categorical_training_features.xlsx"
# encoders picle
encoders_pickle_file = 'encoders_pickle_file.pickle'
# files names definitions
client_info_file = "02-Clients database (credit model).xlsx"
latest_client_info_file = "latest_clients_info.xlsx"
latest_share_of_wallet_file = "latest_latest_share_of_wallet.xlsx"
share_of_wallet_file = "03-Share of wallet (PEC dataset).xlsx"
all_training_features = pd.read_excel(root_folder + model_validation_folder + all_training_features_file)
########################
## Client info (credit model) dataset
########################
client_info =\
    pd.read_excel(root_folder + data_folder + raw_data_folder + client_info_file,
                  sheet_name= "data")
print(client_info.shape)
latest_client_info =\
    client_info.loc[client_info.groupby('ID Group Mck').Safra.idxmax(),
                    np.isin(client_info.columns,
                              np.concatenate([np.array(['Safra', 'ID Group Mck']),
                                              np.array(all_training_features[0].values)]))]

print(latest_client_info.shape)
print(latest_client_info.columns)
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

print(share_of_wallet.shape)
latest_share_of_wallet = \
    share_of_wallet.loc[share_of_wallet.groupby('COD GRUPO MCK').Safra_ShareOfWallet.idxmax(),
                        np.isin(share_of_wallet.columns,
                                   np.concatenate([np.array(['Safra_ShareOfWallet', 'COD GRUPO MCK']),
                                                   np.array(all_training_features[0].values)]))]

print(latest_share_of_wallet.shape)
print(latest_share_of_wallet.columns)
##
numeric_training_features = \
    pd.read_excel(root_folder + model_validation_folder + numeric_training_features_file)
categorical_training_features = \
    pd.read_excel(root_folder + model_validation_folder + categorical_training_features_file)
##
numeric_training_features_sw = \
    latest_share_of_wallet.columns[np.isin(latest_share_of_wallet.columns,
                                           numeric_training_features)]
categorical_training_features_sw = \
    latest_share_of_wallet.columns[np.isin(latest_share_of_wallet.columns,
                                           categorical_training_features)]
if(len(numeric_training_features_sw)>0):
    for variable in numeric_training_features_sw:
        latest_share_of_wallet[variable] = latest_share_of_wallet[variable].astype(float)
        latest_share_of_wallet.loc[latest_share_of_wallet[variable].isna(), variable] = \
            latest_share_of_wallet.loc[~latest_share_of_wallet[variable].isna(),variable].mean()

with open(root_folder + model_object_folder + encoders_pickle_file, "rb") as pickle_encoders_file:
    list_label_enc = pickle.load(pickle_encoders_file)
if len(categorical_training_features_sw) > 0 :
    for column in categorical_training_features_sw:
        le = list(compress(list_label_enc, column == categorical_training_features[0].values))[0]
        latest_client_info[column] = latest_client_info[column].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        latest_client_info[column] = le.transform(latest_client_info[column])
##
latest_client_info.loc[latest_client_info['logPD'].isna(),'logPD'] =\
    latest_client_info.loc[~latest_client_info['logPD'].isna(),'logPD'].min()

numeric_training_features_cl = \
    latest_client_info.columns[np.isin(latest_client_info.columns,
                                       numeric_training_features)]
categorical_training_features_cl = \
    latest_client_info.columns[np.isin(latest_client_info.columns,
                                       categorical_training_features)]
if(len(numeric_training_features_cl)>0):
    for variable in numeric_training_features_cl:
        latest_client_info[variable] = latest_client_info[variable].astype(float)
        latest_client_info.loc[latest_client_info[variable].isna(), variable] = \
            latest_client_info.loc[~latest_client_info[variable].isna(),variable].mean()
#####
# categorical variables
#####
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

variables_relationship_cl = \
    latest_client_info.columns[np.isin(latest_client_info.columns,
                                       variables_relationship)]
if(len(variables_relationship_cl)>0):
    for variable in variables_relationship_cl:
        latest_client_info.loc[latest_client_info[variable]==False,variable] = 0
        latest_client_info.loc[latest_client_info[variable]==True,variable] = 1
        latest_client_info.loc[latest_client_info[variable].isna(),variable] = 2
# complexity variables
complexity_relationship =\
    ['MAX_Complexity',
     'MAX_Complexity_1',
     'MAX_Complexity_3',
     'MAX_Complexity_6',
     'MAX_Complexity_12']
complexity_relationship_cl = \
    latest_client_info.columns[np.isin(latest_client_info.columns,
                                       complexity_relationship)]
# complexity variables
if(len(complexity_relationship_cl)>0):
    for variable in complexity_relationship_cl:
        latest_client_info.loc[:,variable] = latest_client_info.loc[:,variable].astype(int)
# transform categorical variable using Label encoder
if len(categorical_training_features_cl) > 0 :
    for column in categorical_training_features_cl:
        le = list(compress(list_label_enc, column == categorical_training_features[0].values))[0]
        latest_client_info[column] = latest_client_info[column].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        latest_client_info[column] = le.transform(latest_client_info[column])
##
latest_client_info.to_excel(root_folder + data_folder + implementation_dataset_folder + latest_client_info_file, sheet_name= "data")
latest_share_of_wallet.to_excel(root_folder + data_folder + implementation_dataset_folder + latest_share_of_wallet_file, sheet_name= "data")
###
input_data_for_kNN = 'input_data_for_kNN.pickle'
client_hist_spread_file = 'client_hist_spread.pickle'
# read dataset file
with open(root_folder + model_object_folder + input_data_for_kNN, "rb") as pickle_kNN_input_data_file:
    input_data = pickle.load(pickle_kNN_input_data_file)
#####
# data for testing
historical_spread_client = input_data[['Cod Grupo McK', 'Margem USD', 'Volume USD', 'Spread bps']]
##
historical_spread_client_gb = \
    historical_spread_client.groupby('Cod Grupo McK').agg(
        {'Margem USD':'sum',
         'Volume USD':'sum',
         'Spread bps':['min', 'max']}).reset_index()

historical_spread_client_gb.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in historical_spread_client_gb.columns]
##
historical_spread_client_gb['Spread bps_mean'] =\
    10000*historical_spread_client_gb['Margem USD_sum'].values/historical_spread_client_gb['Volume USD_sum'].values
historical_spread_client_gb.drop(columns = ['Margem USD_sum',
                                            'Volume USD_sum'], inplace = True)
###
with open(root_folder + model_object_folder + client_hist_spread_file, "wb") as pickle_client_hist_spread:
    pickle.dump(historical_spread_client_gb, pickle_client_hist_spread)

pd.DataFrame(historical_spread_client_gb).to_excel(root_folder + data_folder + implementation_dataset_folder + "historical_spread_client_gb.xlsx")
