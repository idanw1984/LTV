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
CLUSTER_NUM = 3
MAX_ITER = 600
N_INIT = 100
COLUMNS_FOR_KMEANS_D30= ['Transc_7d','Net_IAP_7d','Transc_14d','Net_IAP_14d','Transc_30d','Net_IAP_30d']
COLUMNS_FOR_KMEANS_D14= ['Transc_7d','Net_IAP_7d','Transc_14d','Net_IAP_14d']
COLUMNS_FOR_KMEANS_D7= ['Transc_7d','Net_IAP_7d']
COLUMNS_FOR_LR_D7= ['Transc_7d','Net_IAP_7d']
COLUMNS_FOR_Xgboost_D7_Y = ['Net_IAP_30d']
COLUMNS_FOR_Xgboost_D14_Y = ['Net_IAP_60d']
COLUMNS_FOR_Xgboost_D30_Y = ['Net_IAP_90d']
# FEATURES_FOR_XGBOOST = ['Transc_','Net_IAP_','reward_','level_completed_','level_started_','vm_transaction_','activity_']
FEATURES_FOR_XGBOOST = ['Transc_','Net_IAP_','level_completed_','vm_transaction_','activity_']
DAYS_TO_PREDICT = {'7d':'30d','14d':'60d','30d':'90d'}
PARAMS = {"objective":["reg:linear"],'colsample_bytree': [i / 10.0 for i in range(6, 9)],'learning_rate': [i/100 for i in range(1,20,4)],'n_estimators': [800, 900, 1000],
                'max_depth': range(3, 6), 'reg_alpha':[1e-5, 1e-2, 0.1, 1 ]}
CV = 3
##################################################################################

now = datetime.datetime.now()   # get today datetime

#### credentials for bigquery
credentials = service_account.Credentials.from_service_account_file(
    'C:/Json/cooladata-c-ilyon-373980a51847.json')
project_id = 'cooladata-c-ilyon'
client = bigquery.Client(credentials= credentials,project=project_id)
##############################################################################

### sql query to get training data
sql ="""
 SELECT * FROM `cooladata-c-ilyon.project_121656_prod_dataset.pltv_train_data`
  """
#################################################################

dict_scaler ={}
df_test = client.query(sql, project=project_id).to_dataframe()  # get train data
apps = df_test['bundle_id'].unique()

for i in range(len(apps)):
    for day in DAYS:
     dict_scaler[('scaler_'+day,apps[i])] = StandardScaler()

dict_std = {}
for i in  range(len(apps)):
    for day in DAYS:
     dict_std[(day, apps[i])] = dict_scaler[('scaler_'+day, apps[i])].fit_transform(df_test[(df_test['vetek_'+day]==1) & (df_test['bundle_id']==apps[i])][COLUMNS_FOR_KMEANS_D7])


######################################Training  Phase #########################################

kmeans_dict = {}
for i in  range(len(apps)):
    for day in DAYS:
        kmeans_dict[(day,apps[i])] = KMeans(n_clusters=CLUSTER_NUM, init='k-means++', max_iter=MAX_ITER, n_init=N_INIT, random_state=0)
        kmeans_dict[('Train_Kmeans_'+day+'_fit', apps[i])] = kmeans_dict[(day,apps[i])].fit(dict_std[(day, apps[i])])
        kmeans_dict[('Train_Kmeans_'+day+'_predict', apps[i])] = kmeans_dict[('Train_Kmeans_'+day+'_fit', apps[i])].fit_predict(dict_std[(day, apps[i])])
        kmeans_dict[('train_data_'+day+'_with_pred', apps[i])] = df_test[(df_test['vetek_'+day]==1) & (df_test['bundle_id']==apps[i])].copy()
        kmeans_dict[('train_data_'+day+'_with_pred', apps[i])]['cluster'] = kmeans_dict[('Train_Kmeans_'+day+'_predict', apps[i])]

columns_list = {}
for i,day in zip(range(len(DAYS)),DAYS):
    for app in apps:
        list_col = list()
        for feature in FEATURES_FOR_XGBOOST:
            j=i
            while(j>=0):
                list_col.append(feature+DAYS[j])
                j=j-1
        for cluster in kmeans_dict[('train_data_'+day+'_with_pred', app)]['cluster'].unique():
            list_col.append('cluster_'+str(cluster))
        for campagin_seg in df_test['c_seg_id'].unique():
            list_col.append('c_seg_id_'+str(campagin_seg))
        for netsource_id in df_test['netsource_id'].unique():
            list_col.append('netsource_id_'+str(netsource_id))
        columns_list[(day,app)] = list_col.copy()


train_data_matrix = {}
Xgboost_model_list = {}
for app in apps:
    for day in DAYS:
        train_data_matrix[('X',day,app)],train_data_matrix[('y',day,app)] = pd.get_dummies(kmeans_dict[('train_data_'+day+'_with_pred', app)],columns = ['cluster','c_seg_id','netsource_id']),pd.get_dummies(kmeans_dict[('train_data_'+day+'_with_pred', app)],columns = ['cluster','c_seg_id','netsource_id'])[['Customer','Net_IAP_'+DAYS_TO_PREDICT[day]]]
        train_data_matrix[('X',day,app)] = train_data_matrix[('X',day,app)].loc[:,columns_list[(day,app)]+list(['Customer','install_date','bundle_id','campaign'])]
        train_data_matrix[('X',day,app)] = train_data_matrix[('X',day,app)].set_index(['Customer','install_date','bundle_id','campaign'])
        train_data_matrix[('y',day,app)] = train_data_matrix[('y',day,app)].set_index(['Customer'])
        train_data_matrix[('X_train',day,app)] ,train_data_matrix[('X_test',day,app)] ,train_data_matrix[('y_train',day,app)] , train_data_matrix[('y_test',day,app)] = train_test_split(train_data_matrix[('X',day,app)], train_data_matrix[('y',day,app)] , test_size=0.3, random_state=123)
        Xgboost_model_list[('xg_reg_Train_model',day,app)] = XGBRegressor(n_jobs=1, objective='reg:squarederror', scoring='mae', bootstrap=True)
        Xgboost_model_list[('xg_grid_Train_model', day, app)] = GridSearchCV(estimator=Xgboost_model_list[('xg_reg_Train_model',day,app)], param_grid=PARAMS, cv=CV, refit=True,n_jobs=1)
        Xgboost_model_list[('xg_fit_Train_model', day, app)] = Xgboost_model_list[('xg_grid_Train_model', day, app)].fit(train_data_matrix[('X_train',day,app)],train_data_matrix[('y_train',day,app)])
        Xgboost_model_list[('xg_predictions_on_Test', day, app)] = Xgboost_model_list[('xg_fit_Train_model', day, app)].predict(train_data_matrix[('X_test',day,app)])
        Xgboost_model_list[('xg_RMSE_score_on_Test', day, app)] = np.sqrt(mean_squared_error(train_data_matrix[('y_test',day,app)],Xgboost_model_list[('xg_predictions_on_Test', day, app)]))
        Xgboost_model_list[('xg_MAE_score_on_Test', day, app)] = mean_absolute_error(train_data_matrix[('y_test',day,app)],Xgboost_model_list[('xg_predictions_on_Test', day, app)])

#################### Xgboost Model dump #########################
today_date = now.strftime("%d-%m-%Y")
# file_string = 'C:/Models/CurrentModel/XgboostCurrentModel'
# file = open(file_string, 'wb')
file_string2 = 'C:/Models/PreviousModels/XgboostCurrentModel_'+today_date
file2 = open(file_string2, 'wb')
# pickle.dump(Xgboost_model_list, file)
pickle.dump(Xgboost_model_list, file2)
# close the file
# file.close()
file2.close()




############### Kmeans clustering dump
file_string = 'C:/Models/CurrentModel/KmeansCurrentModel'
file = open(file_string, 'wb')
file_string2 = 'C:/Models/PreviousModels/KmeansCurrentModel_'+today_date
file2 = open(file_string2, 'wb')
pickle.dump(kmeans_dict, file)
pickle.dump(kmeans_dict, file2)
# close the file
file.close()
file2.close()

