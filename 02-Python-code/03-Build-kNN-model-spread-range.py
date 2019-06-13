from IPython import get_ipython;
get_ipython().magic('reset -sf')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
from sklearn.decomposition import PCA
import scipy
import xgboost as xgb
import random

# folders names definitions
root_folder = 'C:/Users/Pablo Andrade/Box Sync/FX Pricing Model/09-Handover/'
model_validation_folder = '05-Model validation/'
model_object_folder = '06-Models objects (pickle)/'
# files names definitions
features_importance_xgboost_csv = 'varImp-XGBoost.csv'
KNN_test_results_file = "KNN-test-results.xlsx"
xgboost_reference_price_model = 'xgboost_reference_price_model.pickle'
# knn file
input_data_for_kNN = 'input_data_for_kNN.pickle'
# pickle files for kNN
scaler_object_for_rNN_file = 'scaler_object_for_rNN.pickle'
pca_object_for_rNN_file = 'pca_object_for_rNN.pickle'
data_distance_calculation_for_rNN_file = 'data_distance_calculation_for_rNN.pickle'
data_radius_for_rNN_file = 'data_radius_for_rNN.pickle'
target_for_rNN_file = "target_for_rNN.pickle"
col_correlation_spread_file = 'col_correlation_spread.pickle'

# read dataset file
with open(root_folder + model_object_folder + input_data_for_kNN, "rb") as pickle_kNN_input_data_file:
    input_data = pickle.load(pickle_kNN_input_data_file)
#####
features_importance =\
    pd.read_csv(root_folder + model_validation_folder + features_importance_xgboost_csv, sep=';', decimal=',')
#####
# data for testing
margin_client_datetime = input_data[['Cod Grupo McK','datetime', 'Margem USD']]
volume_client_datetime = input_data[['Cod Grupo McK','datetime', 'Volume USD']]
spread_client_datetime = input_data[['Cod Grupo McK','datetime', 'Spread bps']]
margin_client_datetime.sort_values(by=['Cod Grupo McK','datetime'], inplace=True)
volume_client_datetime.sort_values(by=['Cod Grupo McK','datetime'], inplace=True)
spread_client_datetime.sort_values(by=['Cod Grupo McK','datetime'], inplace=True)
##
cummargin_client_datetime =\
    margin_client_datetime.groupby(by=['Cod Grupo McK','datetime']).sum().groupby(level=[0]).cumsum().reset_index()
cumvolume_client_datetime = \
    volume_client_datetime.groupby(by=['Cod Grupo McK','datetime']).sum().groupby(level=[0]).cumsum().reset_index()
##
spread_min_client_datetime =\
    spread_client_datetime.groupby(by=['Cod Grupo McK', 'datetime']).min().groupby(level=[0]).cummin().reset_index()
spread_max_client_datetime =\
    spread_client_datetime.groupby(by=['Cod Grupo McK', 'datetime']).sum().groupby(level=[0]).cummax().reset_index()
##
avg_spread = cummargin_client_datetime[['Cod Grupo McK','datetime']]
avg_spread['client_hist_avg_spread'] =\
    10000*cummargin_client_datetime['Margem USD'].values/cumvolume_client_datetime['Volume USD'].values
avg_spread['client_hist_min_spread'] =\
    spread_min_client_datetime['Spread bps'].values
avg_spread['client_hist_max_spread'] =\
    spread_max_client_datetime['Spread bps'].values
##
print(input_data.shape)
input_data =\
    pd.merge(input_data,
             avg_spread,
             how='left',
             left_on=['Cod Grupo McK','datetime'],
             right_on = ['Cod Grupo McK','datetime'])
print(input_data.shape)
# keep only variables used by XGBoost
input_data = input_data.dropna(subset=features_importance['Feature'].unique())
#####
# Split the data into training and testing sets
targets = input_data['Spread bps']
input_data.drop(columns = ['Cod Grupo McK', 'Spread bps','Margem USD'], inplace = True)
X_train, X_test, y_train, y_test = train_test_split(input_data, targets, test_size=0.20, random_state=42)
X_train = X_train.sort_values(by="datetime")
######
# read XGBoost model from disk
with open(root_folder + model_object_folder + xgboost_reference_price_model, "rb") as pickle_output_file:
    xgb_model = pickle.load(pickle_output_file)
######
volume_testds = X_test['Volume USD']
client_hist_avg_spread_testds = X_test['client_hist_avg_spread']
client_hist_min_spread_testds = X_test['client_hist_min_spread']
client_hist_max_spread_testds = X_test['client_hist_max_spread']
datetime_testds = X_test['datetime']
#####
predictions_testds = \
    xgb_model.predict(
        xgb.DMatrix(X_test.loc[:,
                    ~np.isin(X_test.columns,['datetime',
                                              'client_hist_avg_spread',
                                              'client_hist_min_spread',
                                              'client_hist_max_spread'])]))
####################
n_components_selected = 5
scaler = preprocessing.StandardScaler()
pca = PCA(n_components=n_components_selected)
####################
X_train_selected_vars = X_train.loc[:, np.isin(X_train.columns, features_importance['Feature'].unique())]
X_train_scaled = scaler.fit(X_train_selected_vars)
X_train_scaled = scaler.transform(X_train_selected_vars)
##
X_test_selected_vars = X_test.loc[:, np.isin(X_test.columns, features_importance['Feature'].unique())]
X_test_scaled = scaler.transform(X_test_selected_vars)
####################
SEED = 4449
random.seed(SEED)
s = random.sample(range(0, X_train_scaled.shape[0]), 100)
sample1 = X_train_scaled[s,:]
spread1 = y_train.iloc[s].values
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
    distances_spread.shape = (1,distances_spread.shape[0]*distances_spread.shape[1])
    distances_col.shape = (1,distances_col.shape[0]*distances_col.shape[1])
    col_correlation_spread.append(np.corrcoef(distances_col[0,:], distances_spread[0,:])[0,1])
####################
X_train_scaled = X_train_scaled[:,[x[0] for x in enumerate(col_correlation_spread) if x[1] > 0.0]]
X_test_scaled = X_test_scaled[:,[x[0] for x in enumerate(col_correlation_spread) if x[1] > 0.0]]
####################
pca.fit(X_train_scaled)
X_train_scaled_pca = pca.transform(X_train_scaled)
X_test_scaled_pca = pca.transform(X_test_scaled)
###
df_train_scaled_pca = pd.DataFrame(data=X_train_scaled_pca)
df_test_scaled_pca = pd.DataFrame(data=X_test_scaled_pca)
###
distances = scipy.spatial.distance.cdist(df_test_scaled_pca,
                                         df_train_scaled_pca,
                                         metric='euclidean')
###
r = 2.0
count_missing_fin = 0
range_spread_list_testds = np.empty((0, distances.shape[1]), float)
number_of_neighbors_testds = []
range_spread_list_testds = y_train.values * ((distances <= r) & (distances>0))
range_spread_list_testds = pd.DataFrame(range_spread_list_testds)
print(range_spread_list_testds.shape)
print(predictions_testds.shape)
range_spread_list_testds_rowsum = range_spread_list_testds.sum(axis=1)
y_test = y_test.iloc[range_spread_list_testds_rowsum.values>0]
volume_testds = volume_testds.values[range_spread_list_testds_rowsum.values>0]
datetime_testds = datetime_testds.values[range_spread_list_testds_rowsum.values>0]
predictions_testds = predictions_testds[range_spread_list_testds_rowsum.values>0]
client_hist_avg_spread_testds = client_hist_avg_spread_testds.values[range_spread_list_testds_rowsum.values>0]
client_hist_max_spread_testds = client_hist_max_spread_testds.values[range_spread_list_testds_rowsum.values>0]
client_hist_min_spread_testds = client_hist_min_spread_testds.values[range_spread_list_testds_rowsum.values>0]
range_spread_list_testds = range_spread_list_testds.loc[range_spread_list_testds_rowsum.values>0,:]
print(range_spread_list_testds.shape)
print(predictions_testds.shape)
mask_values = (range_spread_list_testds == 0)
##
nneighbors= (range_spread_list_testds!=0).sum(axis=1)

df_range_spread_financial_percentiles =\
    np.nanpercentile(a=range_spread_list_testds.mask(mask_values),
                     q=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95],
                     axis=1)
####
predictions_and_range = \
    pd.DataFrame({'datetime':datetime_testds,
                  'reference_price':predictions_testds,
                  'percentile_05':df_range_spread_financial_percentiles[0,:],
                  'percentile_10':df_range_spread_financial_percentiles[1,:],
                  'percentile_15':df_range_spread_financial_percentiles[2,:],
                  'percentile_20':df_range_spread_financial_percentiles[3,:],
                  'percentile_25':df_range_spread_financial_percentiles[4,:],
                  'percentile_30':df_range_spread_financial_percentiles[5,:],
                  'percentile_35':df_range_spread_financial_percentiles[6,:],
                  'percentile_40': df_range_spread_financial_percentiles[7,:],
                  'percentile_45': df_range_spread_financial_percentiles[8,:],
                  'percentile_50': df_range_spread_financial_percentiles[9,:],
                  'percentile_55': df_range_spread_financial_percentiles[10,:],
                  'percentile_60': df_range_spread_financial_percentiles[11,:],
                  'percentile_65': df_range_spread_financial_percentiles[12,:],
                  'percentile_70': df_range_spread_financial_percentiles[13,:],
                  'percentile_75': df_range_spread_financial_percentiles[14,:],
                  'percentile_80': df_range_spread_financial_percentiles[15,:],
                  'percentile_85': df_range_spread_financial_percentiles[16,:],
                  'percentile_90': df_range_spread_financial_percentiles[17,:],
                  'percentile_95': df_range_spread_financial_percentiles[18,:],
                  'real_spread':y_test,
                  'volume USD':volume_testds,
                  'nneighbors':nneighbors.values,
                  'client_hist_avg_spread':client_hist_avg_spread_testds,
                  'client_hist_min_spread': client_hist_min_spread_testds,
                  'client_hist_max_spread': client_hist_max_spread_testds})

predictions_and_range.to_excel(root_folder + model_validation_folder + KNN_test_results_file)

## save data and objects for kNN
with open(root_folder + model_object_folder + scaler_object_for_rNN_file, "wb") as pickle_kNN_scaler_object_file:
    pickle.dump(scaler, pickle_kNN_scaler_object_file)
with open(root_folder + model_object_folder + pca_object_for_rNN_file, "wb") as pickle_kNN_pca_object_file:
    pickle.dump(pca, pickle_kNN_pca_object_file)
with open(root_folder + model_object_folder + data_distance_calculation_for_rNN_file, "wb") as pickle_kNN_distance_data_object_file:
    pickle.dump(df_train_scaled_pca, pickle_kNN_distance_data_object_file)
with open(root_folder + model_object_folder + target_for_rNN_file, "wb") as pickle_target_for_rNN_file:
    pickle.dump(y_train, pickle_target_for_rNN_file)
with open(root_folder + model_object_folder + data_radius_for_rNN_file, "wb") as pickle_kNN_distance_data_object_file:
    pickle.dump(r, pickle_kNN_distance_data_object_file)
with open(root_folder + model_object_folder + col_correlation_spread_file, "wb") as pickle_col_correlation_spread_file:
    pickle.dump(col_correlation_spread, pickle_col_correlation_spread_file)
