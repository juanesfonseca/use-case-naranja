
root_folder = 'C:/Users/Pablo Andrade/Documents/2019-01-Jan/01-BTU-FX-Pricing/09-Hand-over/'
implementation_data_folder = '01-Data/03-Implementation/'
implementation_market_data_file = '01-market_data.csv'
implementation_external_data_file = '02-external_data.csv'
implementation_past_trades_data_file = '03-past_trades_data.csv'
implementation_client_info_data_file = '04-client_info_data.csv'
implementation_share_of_wallet_data_file = '05-share_of_wallet_data.csv'

market_variables = ['US_Govt_Bond_10Y_Open', 'US_Govt_Bond_10Y_High',
       'US_Govt_Bond_10Y_Low', 'US_Govt_Bond_10Y_Close',
       'US_Govt_Bond_5Y_Open', 'US_Govt_Bond_5Y_High',
       'US_Govt_Bond_5Y_Low', 'US_Govt_Bond_5Y_Close',
       'Brazil_Bond_10Y_Open', 'Brazil_Bond_10Y_High',
       'Brazil_Bond_10Y_Low', 'Brazil_Bond_10Y_Close',
       'Brazil_Bond_5Y_Open', 'Brazil_Bond_5Y_High', 'Brazil_Bond_5Y_Low',
       'Brazil_Bond_5Y_Close', 'JPMEMCI_Open', 'JPMEMCI_High',
       'JPMEMCI_Low', 'JPMEMCI_Close', 'JPMEMCI_VOLAT_90_Days',
       'JPMEMCI_VOLAT_60_Days', 'JPMEMCI_VOLAT_30_Days',
       'JPMEMCI_VOLAT_15_Days', 'JPMEMCI_VOLAT_1_Day',
       'JPMEMCI_VOLAT_12_Hours', 'JPMEMCI_VOLAT_6_Hours',
       'JPMEMCI_VOLAT_4_Hours', 'JPMEMCI_VOLAT_2_Hours', 'MSCI_Open',
       'MSCI_High', 'MSCI_Low', 'MSCI_Close', 'MSCI_VOLAT_90_Days',
       'MSCI_VOLAT_60_Days', 'MSCI_VOLAT_30_Days', 'MSCI_VOLAT_15_Days',
       'MSCI_VOLAT_1_Day', 'MSCI_VOLAT_12_Hours', 'MSCI_VOLAT_6_Hours',
       'MSCI_VOLAT_4_Hours', 'MSCI_VOLAT_2_Hours', 'B3USD_Open',
       'B3USD_High', 'B3USD_Low', 'B3USD_Close', 'B3USD_VOLAT_90_Days',
       'B3USD_VOLAT_60_Days', 'B3USD_VOLAT_30_Days',
       'B3USD_VOLAT_15_Days', 'B3USD_VOLAT_1_Day', 'B3USD_VOLAT_12_Hours',
       'B3USD_VOLAT_6_Hours', 'B3USD_VOLAT_4_Hours',
       'B3USD_VOLAT_2_Hours', 'BRL_Open', 'BRL_High', 'BRL_Low',
       'BRL_Close', 'BRL_VOLAT_90_Days', 'BRL_VOLAT_60_Days',
       'BRL_VOLAT_30_Days', 'BRL_VOLAT_15_Days', 'BRL_VOLAT_1_Day',
       'BRL_VOLAT_12_Hours', 'BRL_VOLAT_6_Hours', 'BRL_VOLAT_4_Hours',
       'BRL_VOLAT_2_Hours', 'MX_PESO_Open', 'MX_PESO_High', 'MX_PESO_Low',
       'MX_PESO_Close', 'MX_PESO_VOLAT_90_Days', 'MX_PESO_VOLAT_60_Days',
       'MX_PESO_VOLAT_30_Days', 'MX_PESO_VOLAT_15_Days',
       'MX_PESO_VOLAT_1_Day', 'MX_PESO_VOLAT_12_Hours',
       'MX_PESO_VOLAT_6_Hours', 'MX_PESO_VOLAT_4_Hours',
       'MX_PESO_VOLAT_2_Hours']
###
external_variables = ['TXA_REF_CAMBIO_ITAU','TXA_REF_CAMBIO_OPERAC']
###
past_trades_variables = ['Transacoes_Anteriores_Dia']
###
client_info_variables = \
      ['Tipo Capital',
       'SoC', 'Relacionamento com Bradesco', 'Relacionamento com BB',
       'Relacionamento com Santander', 'Relacionamento com City',
       'Relacionamento com Caixa', 'Relacionamento com Safra',
       'Relacionamento com Banri', 'Relacionamento com Outros',
       'Share SV Bradesco', 'Share SV BB', 'Share SV Santander',
       'Share SV City', 'Share SV Caixa', 'Share SV Safra',
       'Share SV Banri', 'Share SV Outros', 'IU Itau', 'IU Mercado',
       'SOR', 'CV_prosp', 'CV_iCEAp_med_VP_Cred', 'RARoC_g_cred',
       'RARoC_g', 'EAD_inicial', 'HURDLE', 'MFB_Cash', 'MFB_Credito',
       'MFB_Fundos', 'MFB_IB', 'MFB_XSell', 'RGO_Cash', 'RGO_Credito',
       'RGO_Fundos', 'RGO_IB', 'RGO_XSell', 'CUSTO_K_Cash',
       'CUSTO_K_Credito', 'CUSTO_K_Fundos', 'CUSTO_K_IB', 'CUSTO_K_XSell',
       'PB_Cash', 'PB_Credito', 'PB_Fundos', 'PB_IB', 'PB_XSell',
       'MFB_Total', 'MFB_ratio', 'MFB_ratio_ex_IB', 'PB_Total',
       'PB_ratio', 'PB_ratio_ex_IB', 'RiscoxLimite', 'Risco CrÃƒÂ©dito',
       'Vendas Liquidas', 'Tam Relativo ao Setor',
       'Relacionamento com Bradesco e BB',
       'Relacionamento com Bancos Pequenos',
       'Relacionamento com Bancos Grandes', 'Share SV Bancos Grandes',
       'logPD', 'logPD_Media_2', 'logPD_Media_12', 'logPD_Tendencia',
       'QTD_Assets', 'Notional_Assets', 'MAX_Complexity', 'QTD_Assets_1',
       'QTD_Assets_3', 'QTD_Assets_6', 'QTD_Assets_12',
       'Notional_Assets_1', 'Notional_Assets_3', 'Notional_Assets_6',
       'Notional_Assets_12', 'MAX_Complexity_1', 'MAX_Complexity_3',
       'MAX_Complexity_6', 'MAX_Complexity_12', 'SoC_Media_2',
       'SoC_Media_12', 'SoC_Tendencia', 'IU Itau_Media_2',
       'IU Itau_Media_12', 'IU Itau_Tendencia', 'IU Mercado_Media_2',
       'IU Mercado_Media_12', 'IU Mercado_Tendencia']
###
share_of_wallet_variables = \
    ['Região',
       '_CAM57_-_CAM57_-ACC', '_CAM57_-_CAM57_-Exportação',
       '_CAM57_-_CAM57_-Financeiro Compra', '_CAM57_-_CAM57_-Importação',
       '_CAM57_-_CAM57_-Financeiro Venda',
       '_CAM57_-_CAM57_-CAM57 Subtotal pronto',
       '_CAM57_-_CAM57_-CAM57 Total', 'ITAU-ACC', 'ITAU-Compra',
       'ITAU-Compras Futuras', 'ITAU-Venda', 'ITAU-Vendas Futuras',
       'ITAU-Subtotal pronto', 'ITAU-Total', 'SHARE_ITAU-ACC',
       'SHARE_ITAU-Compra', 'SHARE_ITAU-Venda', 'SHARE_ITAU-Total',
       'Lost_Deals-ACC', 'Lost_Deals-Compra', 'Lost_Deals-Venda']
###
market_data = pd.concat([datetime.reset_index(drop=True),
                         input_data[market_variables]], axis=1)
market_data.to_csv(root_folder + implementation_data_folder + implementation_market_data_file, sep=';', decimal=',', index=False)
###
external_data = pd.concat([datetime.reset_index(drop=True), input_data[external_variables]], axis=1)
external_data.to_csv(root_folder + implementation_data_folder + implementation_external_data_file, sep=';', decimal=',', index=False)
###
past_trades_data = pd.concat([datetime.reset_index(drop=True),
                              client_IDs,
                              input_data[past_trades_variables]], axis=1)
past_trades_data.to_csv(root_folder + implementation_data_folder + implementation_past_trades_data_file, sep=';', decimal=',', index=False)
###
client_info_data = pd.concat([datetime.reset_index(drop=True),
                              client_IDs,
                              input_data[client_info_variables]], axis=1)
client_info_data.to_csv(root_folder + implementation_data_folder + implementation_client_info_data_file, sep=';', decimal=',', index=False)
###
share_of_wallet_data = pd.concat([datetime.reset_index(drop=True),
                                  client_IDs,
                                  input_data[share_of_wallet_variables]], axis=1)
share_of_wallet_data.to_csv(root_folder + implementation_data_folder + implementation_share_of_wallet_data_file, sep=';', decimal=',', index=False)
###
input_data.columns[~np.isin(input_data.columns,market_variables+external_variables+past_trades_variables+client_info_variables+share_of_wallet_variables)]
