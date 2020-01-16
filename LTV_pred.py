from google.cloud import bigquery
from google.oauth2 import service_account
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data
import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter
from lifetimes.utils import calibration_and_holdout_data
import matplotlib as plt
import dill
# job_config = bigquery.QueryJobConfig()
# dataset_id = 'project_121654_prod_dataset'
#
# credentials = service_account.Credentials.from_service_account_file(
#     'C:/Json/cooladata-c-ilyon-373980a51847.json')
# project_id = 'cooladata-c-ilyon'
# client = bigquery.Client(credentials= credentials,project=project_id)
# query_job = client.query("""
#   select customer_user_id,date,event_value_in_usd from `cooladata-c-ilyon.project_121654_prod_dataset.Table_inapp_purchase_ilyon_shooter_ios`
# where install_date = '2019-10-01' and date_diff(date,install_date,day)<=6
#   """,job_config=job_config)
# sql ="""
#   select customer_user_id,date,event_value_in_usd from `cooladata-c-ilyon.project_121654_prod_dataset.Table_inapp_purchase_ilyon_shooter_ios`
# where install_date = '2019-10-01' and date_diff(date,install_date,day)<=6
#   """
# df_test = client.query(sql, project=project_id).to_dataframe()
df_ltv_G= pd.read_csv(r'C:\Users\idan_w\PycharmProjects\LTV\Datasets\20191106\df_ltv_test_tcpa.csv')
# df_ltv_G = pd.read_csv('df_ltv_project_T2.csv')
df_test_G = summary_data_from_transaction_data(df_ltv_G,customer_id_col='Customer',datetime_col='Date',monetary_value_col='IAP',freq='D') # test
# df_test_B = summary_data_from_transaction_data(df_ltv_B,customer_id_col='Customer',datetime_col='Date',monetary_value_col='IAP',freq='D') # test
df = pd.read_csv(r'C:\Users\idan_w\PycharmProjects\LTV\Datasets\20191106\df_ltv_us_all.csv') #df for training

df_ready_for_RFM = summary_data_from_transaction_data(df,customer_id_col='Customer',datetime_col='Date',monetary_value_col='IAP',freq='D')


# similar API to scikit-learn and lifelines.
bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(df_ready_for_RFM['frequency'], df_ready_for_RFM['recency'], df_ready_for_RFM['T'])
# from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases
# plot_calibration_purchases_vs_holdout_purchases(bgf, df_ready_for_RFM)

from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef = 0.001)
returning_customers_summary = df_ready_for_RFM[df_ready_for_RFM['frequency']>0]
ggf.fit(returning_customers_summary['frequency'],
        returning_customers_summary['monetary_value'])
pred_B = ggf.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    df_test_B['frequency'],
    df_test_B['recency'],
    df_test_B['T'],
    df_test_B['monetary_value'],
    freq= 'D',
    time=1 # months
)
pred_G = ggf.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    df_test_G['frequency'],
    df_test_G['recency'],
    df_test_G['T'],
    df_test_G['monetary_value'],
    freq= 'D',
    time=1 # months
)

# pred = ggf.customer_lifetime_value(
#     bgf, #the model to use to predict the number of future transactions
#     df_test['frequency'],
#     df_test['recency'],
#     df_test['T'],
#     df_test['monetary_value'],
#     time=1 # months
# )

# from lifetimes.plotting import plot_frequency_recency_matrix
#
# plot_frequency_recency_matrix(bgf)
#
# from lifetimes.plotting import plot_probability_alive_matrix
#
# plot_probability_alive_matrix(bgf)


from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf)

df_ready_for_RFM[['monetary_value', 'frequency']].corr()