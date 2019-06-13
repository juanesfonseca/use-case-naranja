from flask import Flask
from flask import request
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import scipy
from flask import json
import os
import threading
import shap

app = Flask(__name__)


@app.route('/')
def index():
    return 'Try reference_price'


@app.route('/feedback', methods=['POST'])
def feedback():
    lof_file_path = "C:/Users/Pablo Andrade/Documents/flask/feedback.log"
    if os.path.exists(lof_file_path):
        with open(lof_file_path, 'r') as feedback_file:
            data = json.load(feedback_file)
        data.append(request.get_json())
        with open(lof_file_path, 'w') as feedback_file:
            feedback_file.write(json.dumps(data))
    else:
        data = [request.get_json()]
        print(data)
        with open(lof_file_path, "w") as feedback_file:
            feedback_file.write(json.dumps(data))
    print('ending ')

    return "Informação atualizada. Obrigado pelo feedback!"

@app.route('/reference_price', methods=['POST'])
def reference_price():
    try:
        # folders names definitions
        root_folder = 'C:/Users/Pablo Andrade/Box Sync/FX Pricing Model/09-Handover/'
        model_object_folder = '06-Models objects (pickle)/'
        scaler_object_for_rNN_file = 'scaler_object_for_rNN.pickle'
        pca_object_for_rNN_file = 'pca_object_for_rNN.pickle'
        data_distance_calculation_for_rNN_file = 'data_distance_calculation_for_rNN.pickle'
        data_radius_for_rNN_file = 'data_radius_for_rNN.pickle'
        xgboost_reference_price_model = 'xgboost_reference_price_model.pickle'
        target_for_rNN_file = 'target_for_rNN.pickle'
        client_hist_spread_file = 'client_hist_spread.pickle'
        col_correlation_spread_file = 'col_correlation_spread.pickle'
        latest_client_info_file = "latest_clients_info.xlsx"
        latest_share_of_wallet_file = "latest_latest_share_of_wallet.xlsx"
        all_training_features_file = "all_training_features.xlsx"
        implementation_dataset_folder = "03-Implementation/"
        model_validation_folder = '05-Model validation/'
        implementation_dataset_folder = "03-Implementation/"
        data_folder = "01-Data/"
        default_values_for_missing_file = "default_values_for_missing.xlsx"
        features_importance_xgboost_csv = 'varImp-XGBoost.csv'

        default_values_for_missing = \
            pd.read_excel(root_folder + data_folder + implementation_dataset_folder + default_values_for_missing_file)

        data_df = pd.read_json(json.dumps(request.get_json()), typ='series').reset_index()
        data_df['sample'] = 0
        data_df = data_df.pivot(columns='index', values=0, index='sample')
        client_id = data_df['Cliente'].values[0]
        data_df.drop(columns = 'Cliente', inplace=True)
        data_df = data_df.fillna(0)
        if data_df['Corretora'].iloc[0]:
            data_df['Corretora'].iloc[0] = 1
        else:
            data_df['Corretora'].iloc[0] = 0
        if str(data_df['Produto'].iloc[0]) == 'COMPRA':
            data_df['Produto'].iloc[0] = 0
        else:
            data_df['Produto'].iloc[0] = 1
        for col in data_df.columns:
            if str(data_df[col].iloc[0]) == "NA":
                data_df[col].iloc[0] = default_values_for_missing.loc[default_values_for_missing['column_name']==col]['value'].values
            data_df[col] = pd.to_numeric(data_df[col])

        latest_share_of_wallet =\
            pd.read_excel(root_folder + data_folder + implementation_dataset_folder + latest_share_of_wallet_file, sheet_name="data")
        latest_client_info = \
            pd.read_excel(root_folder + data_folder + implementation_dataset_folder + latest_client_info_file, sheet_name="data")

        latest_client_info = latest_client_info.loc[latest_client_info['ID Group Mck'] == client_id,:]
        latest_share_of_wallet = latest_share_of_wallet.loc[latest_share_of_wallet['COD GRUPO MCK'] == client_id,:]

        if latest_share_of_wallet.shape[0]<1:
            raise ValueError('No PEC info')
        if latest_client_info.shape[0]<1:
            raise ValueError('No client info')

        print(latest_client_info.shape)
        print(latest_share_of_wallet.shape)

        data_df.index = [0]
        latest_share_of_wallet.index = [0]
        latest_client_info.index = [0]
        data_df = pd.concat([data_df, latest_share_of_wallet, latest_client_info], axis=1)

        data_df = data_df.loc[:,~np.isin(data_df.columns,['Safra', 'ID Group Mck', 'COD GRUPO MCK', 'Safra_ShareOfWallet'])]

        # save the model to disk
        with open(root_folder + model_object_folder + xgboost_reference_price_model, "rb") as pickle_xgb_model_file:
            xgb_model = pickle.load(pickle_xgb_model_file)

        all_training_features =\
            pd.read_excel(root_folder + model_validation_folder + all_training_features_file)

        data_df = data_df[all_training_features[0].values.astype(str)]
        prediction = xgb_model.predict(xgb.DMatrix(data_df))

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(data_df)
        print("Main variables pushing spread up: ")
        print(data_df.columns[shap_values[0,:].argsort()][::-1][0:10])
        print(shap_values[0, shap_values[0, :].argsort()[::-1][0:10]])
        print("Main variables pushing spread down: "),
        print(data_df.columns[shap_values[0,:].argsort()][0:10])
        print(shap_values[0,shap_values[0,:].argsort()[0:10]])

        with open(root_folder + model_object_folder + client_hist_spread_file, "rb") as pickle_client_hist_spread:
            historical_spread_client = pickle.load(pickle_client_hist_spread)

        historical_spread_client =\
            historical_spread_client.loc[historical_spread_client['Cod Grupo McK'] == client_id,:]

        print(historical_spread_client)
        print("Reference price: " + str(prediction[0]))

        # read kNN objects and data
        with open(root_folder + model_object_folder + scaler_object_for_rNN_file, "rb") as pickle_kNN_scaler_object_file:
            scaler = pickle.load(pickle_kNN_scaler_object_file)
        with open(root_folder + model_object_folder + pca_object_for_rNN_file, "rb") as pickle_kNN_pca_object_file:
            pca = pickle.load(pickle_kNN_pca_object_file)
        with open(root_folder + model_object_folder + data_distance_calculation_for_rNN_file, "rb") as pickle_kNN_distance_data_object_file:
            df_train_scaled_pca = pickle.load(pickle_kNN_distance_data_object_file)
        with open(root_folder + model_object_folder + target_for_rNN_file, "rb") as pickle_target_for_rNN_file:
            y_train = pickle.load(pickle_target_for_rNN_file)
        with open(root_folder + model_object_folder + data_radius_for_rNN_file, "rb") as pickle_kNN_distance_data_object_file:
            r = pickle.load(pickle_kNN_distance_data_object_file)
        with open(root_folder + model_object_folder + col_correlation_spread_file, "rb") as pickle_col_correlation_spread_file:
            col_correlation_spread = pickle.load(pickle_col_correlation_spread_file)

        #####
        features_importance = \
            pd.read_csv(root_folder + model_validation_folder + features_importance_xgboost_csv, sep=';', decimal=',')
        data_df_selected_vars = data_df.loc[:, np.isin(data_df.columns, features_importance['Feature'].unique())]
        data_df_scaled = scaler.transform(data_df_selected_vars)
        data_df_scaled = data_df_scaled[:, [x[0] for x in enumerate(col_correlation_spread) if x[1] > 0.0]]
        data_df_scaled_pca = pca.transform(data_df_scaled)
        ##
        distances = scipy.spatial.distance.cdist(data_df_scaled_pca,
                                                 df_train_scaled_pca,
                                                 metric='euclidean')
        range_spread =\
            y_train.values * ((distances <= r) & (distances > 0))
        df_range_spread = pd.DataFrame(range_spread)
        mask_values = (df_range_spread == 0)
        nneighbors = (df_range_spread != 0).sum(axis=1)
        print(nneighbors[0])
        if nneighbors[0] > 0:
            df_range_spread_percentiles = \
                np.nanpercentile(a=df_range_spread.mask(mask_values),
                                 q=[35, 50, 65],
                                 axis=1)

        if historical_spread_client.shape[0]>0:
            if nneighbors[0]>0:
                upperQ = df_range_spread_percentiles[2][0]
                median = df_range_spread_percentiles[1][0]
                lowQ = df_range_spread_percentiles[0][0]

                delta_up = upperQ-median
                delta_down = median-lowQ

                # a Python object (dict):
                result_df = {
                    "Reference_price": str(prediction[0]),
                    "target": str(prediction[0]+(delta_up-delta_down)/2),
                    "max": str(prediction[0]+delta_up),
                    "min": str(prediction[0]-delta_down),
                    "client_max": str(historical_spread_client['Spread bps_max'].iloc[0]),
                    "client_avg": str(historical_spread_client['Spread bps_mean'].iloc[0]),
                    "client_min": str(historical_spread_client['Spread bps_min'].iloc[0]),
                    "nsimilar_trades":str(nneighbors[0])
                }
            else:
                # a Python object (dict):
                result_df = {
                    "Reference_price": str(prediction[0]),
                    "target": 'NO_SIMILAR_TRADES',
                    "max": 'NO_SIMILAR_TRADES',
                    "min": 'NO_SIMILAR_TRADES',
                    "client_max": str(historical_spread_client['Spread bps_max'].iloc[0]),
                    "client_avg": str(historical_spread_client['Spread bps_mean'].iloc[0]),
                    "client_min": str(historical_spread_client['Spread bps_min'].iloc[0]),
                    "nsimilar_trades":str(nneighbors[0])
                }
        else:
            if nneighbors[0]>0:
                upperQ = df_range_spread_percentiles[2][0]
                median = df_range_spread_percentiles[1][0]
                lowQ = df_range_spread_percentiles[0][0]

                delta_up = upperQ-median
                delta_down = median-lowQ

                # a Python object (dict):
                result_df = {
                    "Reference_price": str(prediction[0]),
                    "target": str(prediction[0]+(delta_up-delta_down)/2),
                    "max": str(prediction[0]+delta_up),
                    "min": str(prediction[0]-delta_down),
                    "client_max": 'NO_HISTORICAL_INFO',
                    "client_avg": 'NO_HISTORICAL_INFO',
                    "client_min": 'NO_HISTORICAL_INFO',
                    "nsimilar_trades":str(nneighbors[0])
                }
            else:
                # a Python object (dict):
                result_df = {
                    "Reference_price": str(prediction[0]),
                    "target": 'NO_SIMILAR_TRADES',
                    "max": 'NO_SIMILAR_TRADES',
                    "min": 'NO_SIMILAR_TRADES',
                    "client_max": 'NO_HISTORICAL_INFO',
                    "client_avg": 'NO_HISTORICAL_INFO',
                    "client_min": 'NO_HISTORICAL_INFO',
                    "nsimilar_trades":str(nneighbors[0])
                }

        print(result_df)

    except Exception as e:
        print(e)
        # a Python object (dict):
        result_df = {
            "Reference_price": 'Error: ' + str(e),
            "target": 'Error: ' + str(e),
            "max": 'Error: ' + str(e),
            "min": 'Error: ' + str(e),
            "client_max": 'Error: ' + str(e),
            "client_avg": 'Error: ' + str(e),
            "client_min": 'Error: ' + str(e)
        }

    # convert into JSON:
    results_json = json.dumps(result_df)

    return results_json


