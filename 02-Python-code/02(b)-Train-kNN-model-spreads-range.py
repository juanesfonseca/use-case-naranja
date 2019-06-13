import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import xgboost as xgb
import pickle
from scipy.spatial import distance
import scipy
import random
from sklearn.decomposition import PCA

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

predictions = xgb_model.predict(xgb.DMatrix(X_test, label=y_test))
df_measures_knn =\
    pd.DataFrame(columns=['k','n_components','mean_dist_25_75','median_dist_25_75',
                          'total_dist_25_75','mean_variance','median_variance',
                          'total_variance','mean_ratio_variance','median_ratio_variance',
                          'total_ratio_variance','mean_ratio_interquartile','median_ratio_interquartile',
                          'total_ratio_interquartile','df_coverage_real_spread','df_coverage_predicted_spread',
                          'df_dist_prediction_quartiles_to_mean','df_dist_real_quartiles_to_mean',
                          'mean_dist_q_prediction_to_mean','median_dist_q_prediction_to_mean',
                          'total_dist_q_prediction_to_mean','mean_dist_q_real_to_mean',
                          'median_dist_q_real_to_mean','total_dist_q_real_to_mean',
                          'mean_dist_prediction_to_mean', 'median_dist_prediction_to_mean',
                          'total_dist_prediction_to_mean', 'mean_dist_real_to_mean',
                          'median_dist_real_to_mean', 'total_dist_real_to_mean',
                          'mean_dist_25_75_including_pred','median_dist_25_75_including_pred',
                          'total_dist_25_75_including_pred','mean_increase_including_pred'])

# select 100 random sample from test set
n_test_samples = 1000


X_train_knn = X_train
X_train_knn_val = X_train_knn.values
scaler = preprocessing.StandardScaler()
X_train_knn_val_scaled = scaler.fit_transform(X_train_knn_val)
X_train_knn = pd.DataFrame(X_train_knn_val_scaled)

pca = PCA()
pca = pca.fit(X_train_knn)

eigenvalues = pca.explained_variance_
#Plotting the Eigen values
plt.figure()
plt.plot(eigenvalues, 'o')
plt.xlabel('Principal component')
plt.ylabel('Eigen values') #for each component
plt.title('Eigen values')
plt.show()

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o')
plt.xlabel('Principal component')
plt.ylabel('Cummulative Variance (%)') #for each component
plt.title('Cumulative Summation of the Explained Variance')
plt.show()

random_sample = random.sample(range(0, y_test.shape[0]), n_test_samples)
y_test = y_test.iloc[random_sample, :]
predictions = predictions[random_sample]
X_test = X_test.iloc[random_sample, :]

for n_components in [120,110,100,90,80,75,70,60,55,50,45,40,35,25,20]:

    X_train_knn = X_train
    X_train_knn_val = X_train_knn.values
    scaler = preprocessing.StandardScaler()
    X_train_knn_val_scaled = scaler.fit_transform(X_train_knn_val)
    X_train_knn = pd.DataFrame(X_train_knn_val_scaled)

    X_test_knn = X_test
    X_test_knn_val = X_test_knn.values
    X_test_knn_val_scaled = scaler.fit_transform(X_test_knn_val)
    X_test_knn = pd.DataFrame(X_test_knn_val_scaled)

    pca = PCA(n_components=n_components)
    X_train_knn = pca.fit_transform(X_train_knn)
    X_train_knn = pd.DataFrame(data=X_train_knn)

    X_test_knn = pca.transform(X_test_knn)
    X_test_knn = pd.DataFrame(data=X_test_knn)

    for k in [125, 100, 95, 90, 85,
              80, 75, 70, 65, 60, 58, 55, 52,
              50, 48, 45, 42, 40, 38, 35, 32,
              30, 28, 25, 15]:
        distances = scipy.spatial.distance.cdist(X_test_knn, X_train_knn, metric='euclidean')
        distances_order = distances.argsort(axis=1)
        d_closest_cluster = pd.DataFrame(distances_order[:,(k+1):(2*k+1)])
        distances_order = pd.DataFrame(distances_order[:,0:k])

        range_spread_list = np.empty((0,k), float)
        range_spread_cc_list = np.empty((0, k), float)
        for test_sample in range(0, distances_order.shape[0]):
            range_spread = y_train['Spread bps'].iloc[distances_order.iloc[test_sample,:].values]
            range_spread_cc = y_train['Spread bps'].iloc[d_closest_cluster.iloc[test_sample, :].values]
            range_spread_list = \
                np.vstack((range_spread_list, range_spread))
            range_spread_cc_list = \
                np.vstack((range_spread_cc_list, range_spread_cc))
        ############
        df_range_spread = pd.DataFrame(range_spread_list)
        df_range_spread_cc = pd.DataFrame(range_spread_cc_list)

        df_range_spread_75p = df_range_spread.apply(lambda x: np.percentile(x, 75), axis=1)
        df_range_spread_50p = df_range_spread.apply(lambda x: np.percentile(x, 50), axis=1)
        df_range_spread_25p = df_range_spread.apply(lambda x: np.percentile(x, 25), axis=1)

        df_range_spread_75p_include_pred = np.column_stack((df_range_spread_75p, predictions)).max(axis=1)
        df_range_spread_25p_include_pred = np.column_stack((df_range_spread_25p, predictions)).min(axis=1)

        df_coverage_real_spread = \
            sum((df_range_spread_25p < y_test['Spread bps'].values) &
                (df_range_spread_25p < y_test['Spread bps'].values))/n_test_samples
        df_coverage_predicted_spread = \
            sum((df_range_spread_25p < predictions) &
                (df_range_spread_25p < predictions))/n_test_samples

        df_dist_prediction_quartiles_to_mean =\
            abs(predictions-df_range_spread_50p)/(df_range_spread_50p-df_range_spread_25p)
        df_dist_real_quartiles_to_mean = \
            abs(y_test['Spread bps'].values - df_range_spread_50p) / (df_range_spread_50p - df_range_spread_25p)

        df_dist_prediction_to_mean = predictions-df_range_spread_50p
        df_dist_prediction_to_mean.loc[df_dist_prediction_to_mean < 0] = 0

        df_dist_real_to_mean = y_test['Spread bps'].values - df_range_spread_50p
        df_dist_real_to_mean.loc[df_dist_real_to_mean<0] = 0

        df_range_spread_variance = df_range_spread.apply(lambda x: np.var(x), axis=1)

        df_range_spread_75p_cc = df_range_spread_cc.apply(lambda x: np.percentile(x, 75), axis=1)
        df_range_spread_25p_cc = df_range_spread_cc.apply(lambda x: np.percentile(x, 25), axis=1)
        df_range_spread_variance_cc = df_range_spread_cc.apply(lambda x: np.var(x), axis=1)

        df_ratio_variance = df_range_spread_variance/df_range_spread_variance_cc
        df_ratio_interquartile = (df_range_spread_75p-df_range_spread_25p)/(df_range_spread_75p_cc - df_range_spread_25p_cc)

        mean_dist_25_75 = np.average(df_range_spread_75p-df_range_spread_25p)
        median_dist_25_75 = np.median(df_range_spread_75p-df_range_spread_25p)
        total_dist_25_75 = np.sum(df_range_spread_75p-df_range_spread_25p)

        mean_dist_25_75_including_pred =\
            np.average(df_range_spread_75p_include_pred - df_range_spread_25p_include_pred)
        median_dist_25_75_including_pred =\
            np.median(df_range_spread_75p_include_pred - df_range_spread_25p_include_pred)
        total_dist_25_75_including_pred =\
            np.sum(df_range_spread_75p_include_pred - df_range_spread_25p_include_pred)

        mean_increase_including_pred =\
            np.average((df_range_spread_75p_include_pred - df_range_spread_25p_include_pred)-
                       (df_range_spread_75p - df_range_spread_25p))

        mean_variance = np.average(df_range_spread_variance)
        median_variance = np.median(df_range_spread_variance)
        total_variance = np.sum(df_range_spread_variance)

        mean_ratio_variance = np.average(df_ratio_variance)
        median_ratio_variance = np.median(df_ratio_variance)
        total_ratio_variance = np.sum(df_ratio_variance)

        mean_ratio_interquartile = np.average(df_ratio_interquartile)
        median_ratio_interquartile = np.median(df_ratio_interquartile)
        total_ratio_interquartile = np.sum(df_ratio_interquartile)

        mean_dist_q_prediction_to_mean = np.average(df_dist_prediction_quartiles_to_mean.replace([np.inf, -np.inf], np.nan).dropna(how="all"))
        median_dist_q_prediction_to_mean = np.median(df_dist_prediction_quartiles_to_mean.replace([np.inf, -np.inf], np.nan).dropna(how="all"))
        total_dist_q_prediction_to_mean = np.sum(df_dist_prediction_quartiles_to_mean.replace([np.inf, -np.inf], np.nan).dropna(how="all"))

        mean_dist_q_real_to_mean = np.average(df_dist_real_quartiles_to_mean.replace([np.inf, -np.inf], np.nan).dropna(how="all"))
        median_dist_q_real_to_mean = np.median(df_dist_real_quartiles_to_mean.replace([np.inf, -np.inf], np.nan).dropna(how="all"))
        total_dist_q_real_to_mean = np.sum(df_dist_real_quartiles_to_mean.replace([np.inf, -np.inf], np.nan).dropna(how="all"))

        mean_dist_real_to_mean = np.average(df_dist_real_to_mean)
        median_dist_real_to_mean = np.median(df_dist_real_to_mean)
        total_dist_real_to_mean = np.sum(df_dist_real_to_mean)

        mean_dist_prediction_to_mean = np.average(df_dist_prediction_to_mean)
        median_dist_prediction_to_mean = np.median(df_dist_prediction_to_mean)
        total_dist_prediction_to_mean = np.sum(df_dist_prediction_to_mean)

        ############
        df_measures_knn =\
        df_measures_knn.append({'k':k,
                                'n_components':n_components,
                                'mean_dist_25_75': mean_dist_25_75,
                                'median_dist_25_75': median_dist_25_75,
                                'total_dist_25_75': total_dist_25_75,
                                'mean_variance': mean_variance,
                                'median_variance': median_variance,
                                'total_variance': total_variance,
                                'mean_ratio_variance': mean_ratio_variance,
                                'median_ratio_variance': median_ratio_variance,
                                'total_ratio_variance': total_ratio_variance,
                                'mean_ratio_interquartile': mean_ratio_interquartile,
                                'median_ratio_interquartile': median_ratio_interquartile,
                                'total_ratio_interquartile': total_ratio_interquartile,
                                'df_coverage_real_spread':df_coverage_real_spread,
                                'df_coverage_predicted_spread':df_coverage_predicted_spread,
                                'mean_dist_q_prediction_to_mean':mean_dist_q_prediction_to_mean,
                                'median_dist_q_prediction_to_mean':median_dist_q_prediction_to_mean,
                                'total_dist_q_prediction_to_mean':total_dist_q_prediction_to_mean,
                                'mean_dist_q_real_to_mean':mean_dist_q_real_to_mean,
                                'median_dist_q_real_to_mean':median_dist_q_real_to_mean,
                                'total_dist_q_real_to_mean':total_dist_q_real_to_mean,
                                'mean_dist_prediction_to_mean':mean_dist_prediction_to_mean,
                                'median_dist_prediction_to_mean':median_dist_prediction_to_mean,
                                'total_dist_prediction_to_mean':total_dist_prediction_to_mean,
                                'mean_dist_real_to_mean':mean_dist_real_to_mean,
                                'median_dist__real_to_mean':median_dist_real_to_mean,
                                'total_dist_real_to_mean':total_dist_real_to_mean,
                                'mean_dist_25_75_including_pred':mean_dist_25_75_including_pred,
                                'median_dist_25_75_including_pred':median_dist_25_75_including_pred,
                                'total_dist_25_75_including_pred':total_dist_25_75_including_pred,
                                'mean_increase_including_pred':mean_increase_including_pred},
                               ignore_index=True)

# save the training and validation datatsets to disk
with open(root_folder + model_object_folder + df_measures_knn_pickle_file, "wb") as pickle_file:
    pickle.dump(df_measures_knn, pickle_file)

###
plt.figure()
components_list = df_measures_knn['n_components'].unique()
for component in components_list:
    df_measures_knn_comp = df_measures_knn.loc[df_measures_knn['n_components'] == component]
    plt.plot(df_measures_knn_comp['k'],
       100*df_measures_knn_comp['df_coverage_predicted_spread'],'o',
       label=str(component))
    plt.ylim([70,87.5])
    plt.title('Percentage of coverage of the reference spread')
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('% of coverage')
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])
plt.legend()
plt.figure()
for component in components_list:
    df_measures_knn_comp = df_measures_knn.loc[df_measures_knn['n_components'] == component]
    plt.plot(df_measures_knn_comp['k'],
             np.log(df_measures_knn_comp['total_variance']),'o',
             label=str(component))
    plt.title('Total variance intra-cluster (log scale)')
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Variance intra-cluster (log)')
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])
plt.legend()
plt.figure()
for component in components_list:
    df_measures_knn_comp = df_measures_knn.loc[df_measures_knn['n_components'] == component]
    plt.plot(df_measures_knn_comp['k'],
             100 * df_measures_knn_comp['df_coverage_predicted_spread']/np.log(df_measures_knn_comp['total_variance']),'o',
             label=str(component))
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Coverage')
    plt.title('Coverage of reference spread per unit of variance')
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])
plt.legend()
plt.figure()
for component in components_list:
    df_measures_knn_comp = df_measures_knn.loc[df_measures_knn['n_components'] == component]
    plt.plot(df_measures_knn_comp['k'],
             np.log(df_measures_knn_comp['mean_dist_25_75']),'o',
             label=str(component))
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Distance')
    plt.title('Distance inter-quartile')
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])
plt.legend()
plt.figure()
for component in components_list:
    df_measures_knn_comp = df_measures_knn.loc[df_measures_knn['n_components'] == component]
    plt.plot(df_measures_knn_comp['k'],
             100 * df_measures_knn_comp['df_coverage_predicted_spread']/np.log(df_measures_knn_comp['mean_dist_25_75']),'o',
             label=str(component))
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Coverage')
    plt.title('Coverage of reference spread per distance inter-quartile')
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])
plt.legend()
plt.figure()
for component in components_list:
    df_measures_knn_comp = df_measures_knn.loc[df_measures_knn['n_components'] == component]
    plt.plot(df_measures_knn_comp['k'],
             df_measures_knn_comp['mean_ratio_interquartile'],'o',
             label=str(component))
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Coverage')
    plt.title('Mean ratio inter quartile')
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])
plt.legend()

####

from flask import json

lof_file_path = "C:/Users/Pablo Andrade/Documents/flask/feedback.log"

with open(lof_file_path, 'r') as feedback_file:
    data = json.load(feedback_file)

x = {
  "name": "John",
  "age": 30,
  "city": "New York"
}

y = [x,x]
json.dumps(y)