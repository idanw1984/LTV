from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import decimal
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.linear_model import LinearRegression
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.cluster import KMeans
from datetime import datetime
import datetime
import sys
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
import pandas_gbq
import string
import pickle

#constans list:
DAYS = ['7d','14d','30d']
# FEATURES_FOR_XGBOOST = ['Transc_','Net_IAP_','reward_','level_completed_','level_started_','vm_transaction_']
FEATURES_FOR_XGBOOST = ['Transc_','Net_IAP_','level_completed_','vm_transaction_','activity_']
COLUMNS_FOR_KMEANS_30d= ['Transc_7d','Net_IAP_7d','Transc_14d','Net_IAP_14d','Transc_30d','Net_IAP_30d']
COLUMNS_FOR_KMEANS_14d= ['Transc_7d','Net_IAP_7d','Transc_14d','Net_IAP_14d']
COLUMNS_FOR_KMEANS_7d= ['Transc_7d','Net_IAP_7d']


#### credentials for bigquery
credentials = service_account.Credentials.from_service_account_file(
    'C:/Json/cooladata-c-ilyon-373980a51847.json')
project_id = 'cooladata-c-ilyon'
client = bigquery.Client(credentials= credentials,project=project_id)
##############################################################################

### sql query to get prediction data
sql ="""
 SELECT * FROM `cooladata-c-ilyon.project_121656_prod_dataset.pltv_test_data`
  """
#################################################################


df_for_prediction = client.query(sql, project=project_id).to_dataframe()
apps = df_for_prediction['bundle_id'].unique()
file = open('C:/Models/CurrentModel/XgboostCurrentModel', 'rb')
# upload Xgboost model
XgboostModel = pickle.load(file)
# close the file
file.close()
# upload Kmeans model
file = open('C:/Models/CurrentModel/KmeansCurrentModel', 'rb')
KmeansModel = pickle.load(file)
file.close()

prediction_matrix = {}
for app in apps:
    prediction_matrix[('data_7d',app)] =  df_for_prediction[(df_for_prediction['bundle_id']==app)]
    prediction_matrix[('data_14d', app)] = df_for_prediction[ ((df_for_prediction['vetek_7d'] + df_for_prediction['vetek_14d'] + df_for_prediction['vetek_30d']) == 2) & (df_for_prediction['bundle_id'] == app)]
    prediction_matrix[('data_30d', app)] = df_for_prediction[((df_for_prediction['vetek_7d'] + df_for_prediction['vetek_14d'] + df_for_prediction['vetek_30d']) == 3) & (df_for_prediction['bundle_id'] == app)]
    prediction_matrix[('data_7d_std',app)] = KmeansModel[('scaler_7d',app)].transform(prediction_matrix[('data_7d',app)].loc[:,COLUMNS_FOR_KMEANS_7d])
    prediction_matrix[('data_14d_std', app)] = KmeansModel[('scaler_14d', app)].transform(prediction_matrix[('data_14d', app)].loc[:, COLUMNS_FOR_KMEANS_14d])
    prediction_matrix[('data_30d_std', app)] = KmeansModel[('scaler_30d', app)].transform(prediction_matrix[('data_30d', app)].loc[:, COLUMNS_FOR_KMEANS_30d])


columns_list = {}
for i,day in zip(range(len(DAYS)),DAYS):
    for app in apps:
        list_col = list()
        for feature in FEATURES_FOR_XGBOOST:
            j=i
            while(j>=0):
                list_col.append(feature+DAYS[j])
                j=j-1
        for cluster in KmeansModel[('train_data_'+day+'_with_pred', app)]['cluster'].unique():
            list_col.append('cluster_'+str(cluster))
        columns_list[(day,app)] = list_col.copy()




final_df_list = list()
for app in apps:
    for day in DAYS:
        prediction_matrix[('data_kmeans_pred_'+day, app)] = KmeansModel[('Train_Kmeans_'+day+'_fit', app)].predict(prediction_matrix[('data_'+day+'_std', app)])
        prediction_matrix[('data_'+day, app)]['cluster'] = prediction_matrix[('data_kmeans_pred_'+day, app)]
        prediction_matrix[('data_' + day, app)] = pd.get_dummies(prediction_matrix[('data_'+day, app)], columns=['cluster'])
        prediction_matrix[('data_'+day, app)]['preds_'+day] = XgboostModel[('xg_fit_Train_model',day,app)].predict(prediction_matrix[('data_'+day, app)].loc[:,columns_list[(day,app)]])
        final_df_list.append(prediction_matrix[('data_'+day, app)])

df_for_bigquery = pd.concat(final_df_list,ignore_index=True)

job_config = bigquery.LoadJobConfig(
   write_disposition="WRITE_TRUNCATE"
)
job = client.load_table_from_dataframe(
    df_for_bigquery.reset_index(), 'cooladata-c-ilyon.project_121656_prod_dataset.pltv_prediction',job_config=job_config
)
# Wait for the load job to complete.
job.result()
