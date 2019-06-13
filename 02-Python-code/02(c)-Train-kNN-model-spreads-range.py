from IPython import get_ipython;
get_ipython().magic('reset -sf')

from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import xgboost as xgb
import pickle
from scipy.spatial import distance
import scipy
from sklearn.decomposition import PCA
import seaborn as sns

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
features_importance_rf_csv = 'varImp-randomforest.csv'
features_importance_xgboost_csv = 'varImp-XGBoost.csv'
validation_knn_csv = 'validation_knn_csv.csv'
features_correlation_with_target_csv =  'features_correlatoin-with-target.csv'
xgboost_reference_price_model =  'xgboost_reference_price_model.pickle'
features_xgboost_file = 'features_xgboost_file.pickle'
df_measures_knn_pickle_file = 'df_measures_knn.pickle'
X_train_file = 'X_train.pickle'
X_test_file = 'X_test.pickle'
y_train_file = 'y_train.pickle'
y_test_file = 'y_test.pickle'

# save the training and validation datatsets to disk
with open(root_folder + model_object_folder + X_train_file, "rb") as pickle_X_train_file:
    X_train=pickle.load(pickle_X_train_file)
with open(root_folder + model_object_folder + X_test_file, "rb") as pickle_X_test_file:
    X_test=pickle.load(pickle_X_test_file)
with open(root_folder + model_object_folder + y_train_file, "rb") as pickle_y_train_file:
    y_train=pickle.load(pickle_y_train_file)
with open(root_folder + model_object_folder + y_test_file, "rb") as pickle_y_test_file:
    y_test=pickle.load(pickle_y_test_file)

with open(root_folder + model_object_folder + features_xgboost_file, "rb") as pickle_features_file:
    features = pickle.load(pickle_features_file)

# save the model to disk
with open(root_folder + model_object_folder + xgboost_reference_price_model, "rb") as pickle_output_file:
    xgb_model = pickle.load(pickle_output_file)

features_importance =\
    pd.read_csv(root_folder + model_validation_folder + features_importance_xgboost_csv, sep=';', decimal=',')
features_importance =\
    features_importance.sort_values(by=['Importance'], ascending=False)
features_importance_top20 = features_importance['Feature'].iloc[0:20]
features_importance_top = features_importance['Feature']

predictions = xgb_model.predict(xgb.DMatrix(X_test, label=y_test))

n_test_samples = predictions.shape[0]
random_sample = range(0,predictions.shape[0])
#random_sample = random.sample(range(0, y_test.shape[0]), n_test_samples)
n_components_selected = 70
k_selected = 20

y_test = y_test.iloc[random_sample, :]
predictions = predictions[random_sample]
X_test = X_test.iloc[random_sample, :]

X_train_knn = X_train
X_train_knn_val = X_train_knn.values
scaler = preprocessing.StandardScaler()
X_train_knn_val_scaled = scaler.fit_transform(X_train_knn_val)
X_train_knn = pd.DataFrame(X_train_knn_val_scaled)

X_test_knn = X_test
X_test_knn_val = X_test_knn.values
min_max_scaler = preprocessing.StandardScaler()
X_test_knn_val_scaled = min_max_scaler.fit_transform(X_test_knn_val)
X_test_knn = pd.DataFrame(X_test_knn_val_scaled)

pca = PCA(n_components=n_components_selected)
X_train_knn = pca.fit_transform(X_train_knn)
X_train_knn = pd.DataFrame(data=X_train_knn)

X_test_knn = pca.transform(X_test_knn)
X_test_knn = pd.DataFrame(data=X_test_knn)

distances = scipy.spatial.distance.cdist(X_test_knn, X_train_knn, metric='euclidean')
distances_order = distances.argsort(axis=1)
distances_order = pd.DataFrame(distances_order[:,0:k_selected])

range_spread_list = np.empty((0, k_selected), float)
for test_sample in range(0, distances_order.shape[0]):
    range_spread = y_train['Spread bps'].iloc[distances_order.iloc[test_sample, :].values]
    range_spread_list = \
        np.vstack((range_spread_list, range_spread))
############
df_range_spread = pd.DataFrame(range_spread_list)
############

df_range_spread_75p = df_range_spread.apply(lambda x: np.percentile(x, 75), axis=1)
df_range_spread_50p = df_range_spread.apply(lambda x: np.percentile(x, 50), axis=1)
df_range_spread_25p = df_range_spread.apply(lambda x: np.percentile(x, 25), axis=1)

df_range_spread_IQR = df_range_spread_75p - df_range_spread_25p

X_test_low_IQR = X_test.loc[df_range_spread_IQR.values <= 30,
                            np.isin(X_test.columns, features_importance_top)]
X_test_high_IQR = X_test.loc[df_range_spread_IQR.values > 150,
                            np.isin(X_test.columns, features_importance_top)]

column_pvalue =\
    pd.DataFrame(columns=['column','pvalue'])
for col in range(0,X_test_low_IQR.shape[1]):
    column_pvalue = \
        column_pvalue.append({'column':X_test_low_IQR.columns[col],
                              'pvalue':stats.ttest_ind(X_test_low_IQR.iloc[:,col],
                                                   X_test_high_IQR.iloc[:,col])[1]},ignore_index=True)
column_pvalue = column_pvalue.sort_values(by='pvalue')

#X_test_low_IQR = X_test_low_IQR.loc[np.isin(X_test_low_IQR.columns,column_pvalue['column'][0:5]), :]
#X_test_high_IQR = X_test_high_IQR.loc[np.isin(X_test_high_IQR.columns,column_pvalue['column'][0:5]), :]

X_test_low_IQR = X_test_low_IQR.stack().reset_index()
X_test_low_IQR = X_test_low_IQR.loc[:,['level_1', 0]]
X_test_low_IQR['cluster'] = 'low_IQR'
X_test_high_IQR = X_test_high_IQR.stack().reset_index()
X_test_high_IQR = X_test_high_IQR.loc[:,['level_1', 0]]
X_test_high_IQR['cluster'] = 'high_IQR'

X_test_low_IQR = X_test_low_IQR.rename(columns = {'level_1':'Variable',0:'Value'})
X_test_high_IQR = X_test_high_IQR.rename(columns = {'level_1':'Variable',0:'Value'})

column_pvalue = column_pvalue.rename(columns={'column':'Variable'})

X_test_low_IQR = pd.merge(X_test_low_IQR, column_pvalue, how='inner', on='Variable')
X_test_high_IQR = pd.merge(X_test_high_IQR, column_pvalue, how='inner', on='Variable')

X_test_low_high_IQR = X_test_low_IQR.append(X_test_high_IQR)
X_test_low_high_IQR = X_test_low_high_IQR.sort_values(by='pvalue')

#X_test_low_high_IQR_pv = X_test_low_high_IQR.loc[X_test_low_high_IQR['pvalue']<0.01]

variables_list = \
    ['Vendas Liquidas',
     'Brazil_Bond_5Y_Open',
     'US_Govt_Bond_10Y_High',
     'Relacionamento com Bancos Grandes',
      'Tam Relativo ao Setor',
     'SoC_Media_2',
     'TXA_REF_CAMBIO_ITAU',
     'SoC',
     'BRL_Low',
     'BRL_VOLAT_2_Hours',
     'B3USD_Low',
     'logPD_Media_2',
     'MSCI_Close',
     'Auditoria Balanco',
     'logPD_Media_12',
     'JPMEMCI_VOLAT_90_Days',
     'CAM57_ITAU-Compras Futuras'
     'CAM57_SHARE_ITAU-Compra',
     'MAX_Complexity_6',
     'CAM57_ITAU-Compra',
     'Setor 30']

X_test_low_high_IQR_vars =\
    X_test_low_high_IQR.loc[np.isin(X_test_low_high_IQR['Variable'],variables_list), :]

~np.isin(variables_list,X_test_low_high_IQR['Variable'])

idx = 0
for variable in X_test_low_high_IQR_vars['Variable'].unique():
    plt.figure()
    ax = sns.violinplot(x="Value",
                        y="Variable",
                        orient="h",
                        hue="cluster",
                        data=X_test_low_high_IQR_vars.loc[ X_test_low_high_IQR_vars['Variable'] == variable])
    plt.title(variable)
    plt.savefig("C:/Users/Pablo Andrade/Documents/2019-01-Jan/01-BTU-FX-Pricing/06-Figures/" + str(idx) + "-" + variable + ".png" )
    idx = idx + 1

