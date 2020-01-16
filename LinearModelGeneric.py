import numpy as np
from sklearn.linear_model import LinearRegression
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from datetime import datetime
import datetime
import sys
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import pandas_gbq



#constans list:
DAYS = ['D7','D14']
CLUSTER_NUM = 3
MAX_ITER = 600
N_INIT = 100
COLUMNS_FOR_KMEANS_D14= ['Transc_7d','Net_IAP_7d','Transc_14d','Net_IAP_14d']
COLUMNS_FOR_KMEANS_D7= ['Transc_7d','Net_IAP_7d']
COLUMNS_FOR_LR_D7= ['Transc_7d','Net_IAP_7d']
COLUMNS_FOR_LR_D7_Y = ['Net_IAP_30d']
COLUMNS_FOR_LR_D14 = ['Transc_7d','Net_IAP_7d','Transc_14d','Net_IAP_14d']
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

##### sql query to get cost data
cost = """
 SELECT *  FROM `cooladata-c-ilyon.project_121656_prod_dataset.pltv_cost_data_campaign` 
  """

df_cost = client.query(cost, project=project_id).to_dataframe()

#################################################################

### sql query to get training data
inst = """
 SELECT *  FROM `cooladata-c-ilyon.project_121656_prod_dataset.pltv_installers`
  """

df_installers = client.query(inst, project=project_id).to_dataframe()  # get train data

#################################################################

scaler_D7 = StandardScaler()
scaler_D14 = StandardScaler()
dict_scaler ={}
df_test = client.query(sql, project=project_id).to_dataframe()  # get train data
apps = df_test['bundle_id'].unique()
for i in range(len(apps)):
    for day in DAYS:
     dict_scaler[('scaler_'+day,apps[i])] = StandardScaler()

df_test = df_test[df_test['Net_IAP_14d']>0]

dict_std = {}
for i in  range(len(apps)):
    dict_std[('D7', apps[i])] = dict_scaler[('scaler_D7', apps[i])].fit_transform(df_test[(df_test['Net_IAP_7d']>0) & (df_test['bundle_id']==apps[i])][COLUMNS_FOR_KMEANS_D7])
    dict_std[('D14', apps[i])] = dict_scaler[('scaler_D14', apps[i])].fit_transform(df_test[(df_test['bundle_id']==apps[i])][COLUMNS_FOR_KMEANS_D14])


######################################Training  Phase #########################################

kmeans_dict = {}
for i in  range(len(apps)):
    kmeans_dict[('D7',apps[i])] = KMeans(n_clusters=CLUSTER_NUM, init='k-means++', max_iter=MAX_ITER, n_init=N_INIT, random_state=0)
    kmeans_dict[('D14', apps[i])] = KMeans(n_clusters=CLUSTER_NUM, init='k-means++', max_iter=MAX_ITER, n_init=N_INIT, random_state=0)
    kmeans_dict[('Train_Kmeans_D7_fit', apps[i])] = kmeans_dict[('D7',apps[i])].fit(dict_std[('D7', apps[i])])
    kmeans_dict[('Train_Kmeans_D14_fit', apps[i])] = kmeans_dict[('D14', apps[i])].fit(dict_std[('D14', apps[i])])
    kmeans_dict[('Train_Kmeans_D7_predict', apps[i])] = kmeans_dict[('Train_Kmeans_D7_fit', apps[i])].fit_predict(dict_std[('D7', apps[i])])
    kmeans_dict[('Train_Kmeans_D14_predict', apps[i])] = kmeans_dict[('Train_Kmeans_D14_fit', apps[i])].fit_predict(dict_std[('D14', apps[i])])
    kmeans_dict[('train_data_D7_with_pred', apps[i])] = df_test[(df_test['Net_IAP_7d']>0) & (df_test['bundle_id']==apps[i])].copy()
    kmeans_dict[('train_data_D14_with_pred', apps[i])] =  df_test[df_test['bundle_id']==apps[i]]
    kmeans_dict[('train_data_D7_with_pred', apps[i])]['cluster'] = kmeans_dict[('Train_Kmeans_D7_predict', apps[i])]
    kmeans_dict[('train_data_D14_with_pred', apps[i])]['cluster'] = kmeans_dict[('Train_Kmeans_D14_predict', apps[i])]



model_list = {}
for i in range(CLUSTER_NUM):
    for app in apps:
        model_list[(i,app,'D7')] = LinearRegression().fit(X=kmeans_dict[('train_data_D7_with_pred', app)][((kmeans_dict[('train_data_D7_with_pred', app)])['cluster']==i)][COLUMNS_FOR_LR_D7],y=kmeans_dict[('train_data_D7_with_pred', app)][((kmeans_dict[('train_data_D7_with_pred', app)])['cluster']==i)][COLUMNS_FOR_LR_D7_Y])
        model_list[(i,app,'D14')] = LinearRegression().fit(X=kmeans_dict[('train_data_D14_with_pred', app)][((kmeans_dict[('train_data_D14_with_pred', app)])['cluster']==i)][COLUMNS_FOR_LR_D14],y=kmeans_dict[('train_data_D14_with_pred', app)][((kmeans_dict[('train_data_D14_with_pred', app)])['cluster']==i)][COLUMNS_FOR_LR_D7_Y])



# r2_score_list_D7,r2_score_list_D14 = list(),list()
# for i in range(len(model_list_D7)):
#     r2_score_list_D7.insert(i,r2_score(df_test_with_pred_D7[df_test_with_pred_D7['cluster']==i]['Net_IAP_30d'],model_list_D7[i].predict(df_test_with_pred_D7[df_test_with_pred_D7['cluster']==i][COLUMNS_FOR_LR_D7])))
# for i in range(len(model_list_D14)):
#     r2_score_list_D14.insert(i,r2_score(df_test_with_pred_D14[df_test_with_pred_D14['cluster']==i]['Net_IAP_30d'],model_list_D14[i].predict(df_test_with_pred_D14[df_test_with_pred_D14['cluster']==i][COLUMNS_FOR_LR_D14])))

### save coef lists of LinearModel
coef_dict ={}
for i,app,day in zip(range(CLUSTER_NUM),apps,DAYS):
    if(day=='D7'):
     for feature_name,coef_index in zip(COLUMNS_FOR_LR_D7,range(len(COLUMNS_FOR_LR_D7))):
       coef_dict[(i,app,day,feature_name)] = float(model_list[(i,app,day)].coef_[:,coef_index])
    if (day == 'D14'):
        for feature_name, coef_index in zip(COLUMNS_FOR_LR_D14, range(len(COLUMNS_FOR_LR_D14))):
            coef_dict[(i, app, day, feature_name)] = float(model_list[(i, app, day)].coef_[:, coef_index])





training_df_list_D7 =[]
training_df_list_D14 =[]
for i in range(df_test_with_pred_D7['cluster'].nunique()):
    pred_temp = model_list_D7[i].predict(X=df_test_with_pred_D7[(df_test_with_pred_D7['cluster']==i)][COLUMNS_FOR_LR_D7])
    temp_data_D7 =  df_test_with_pred_D7[df_test_with_pred_D7['cluster'] == i].copy()
    temp_data_D7['pred'] = pred_temp[:]
    training_df_list_D7.append(temp_data_D7)
for i in range(df_test_with_pred_D14['cluster'].nunique()):
    pred_temp = model_list_D14[i].predict(X=df_test_with_pred_D14[df_test_with_pred_D14['cluster'] == i][COLUMNS_FOR_LR_D14])
    temp_data_D14 = df_test_with_pred_D14[df_test_with_pred_D14['cluster'] == i].copy()
    temp_data_D14['pred'] = pred_temp[:]
    training_df_list_D14.append(temp_data_D14)
(pd.concat(training_df_list_D7)).to_csv(r'training_df_list_D7_20191209.csv')
(pd.concat(training_df_list_D14)).to_csv(r'training_df_list_D14.csv')
df_test.to_csv(r'C:/Users/idan_w/PycharmProjects/LTV/Outputs/df_test.csv')






######################################End Of Training#########################################






######################################Prediction Phase#########################################


sql_for_pred ="""
 SELECT * FROM `cooladata-c-ilyon.project_121656_prod_dataset.pltv_test_data_reg_model`
 where bundle_id='com.ilyondynamics.bubbleShooterClassic'
  """
df_for_pred = client.query(sql_for_pred, project=project_id).to_dataframe()

df_for_pred_D7,df_for_pred_D14 = df_for_pred[(df_for_pred['Transc_7d']>0)&(df_for_pred['vetek_d14']==0)].copy(),df_for_pred[df_for_pred['vetek_d14']==1].copy()

df_for_pred_std_D7,df_for_pred_std_D14 = scaler_D7.transform(df_for_pred_D7.loc[:,COLUMNS_FOR_KMEANS_D7]),scaler_D14.transform(df_for_pred_D14.loc[:,COLUMNS_FOR_KMEANS_D14])  # scaling features

kmeans_pred_D7_for_prediction,kmeans_pred_D14_for_prediction = Kmeans_D7.predict(df_for_pred_std_D7),Kmeans_D14.predict(df_for_pred_std_D14)

df_for_pred_D7['cluster'],df_for_pred_D14['cluster'] = kmeans_pred_D7_for_prediction[:],kmeans_pred_D14_for_prediction[:]


if((df_for_pred_D7['cluster'].nunique()<3) |(df_for_pred_D14['cluster'].nunique()<3)):
    filename="C:\Logs_from_models\log"+ str(now)[:10] +".txt"
    f = open(filename, "w+")
    f.write("cluster num for D7 pred: %d\ncluster num for D14: %d" % (df_for_pred_D7['cluster'].nunique(),df_for_pred_D14['cluster'].nunique()))
    f.close()



final_df_list_D7 = []
final_df_list_D14 = []
for i in range(df_for_pred_D7['cluster'].nunique()):
    pred_temp = model_list_D7[i].predict(X=df_for_pred_D7[df_for_pred_D7['cluster']==i][COLUMNS_FOR_LR_D7])
    temp_data_D7 =  df_for_pred_D7[df_for_pred_D7['cluster'] == i].copy()
    temp_data_D7['pred'] = pred_temp[:]
    final_df_list_D7.append(temp_data_D7)
for i in range(df_for_pred_D14['cluster'].nunique()):
    pred_temp = model_list_D14[i].predict(X=df_for_pred_D14[df_for_pred_D14['cluster'] == i][COLUMNS_FOR_LR_D14])
    temp_data_D14 = df_for_pred_D14[df_for_pred_D14['cluster'] == i].copy()
    temp_data_D14['pred'] = pred_temp[:]
    final_df_list_D14.append(temp_data_D14)
final_df_list = final_df_list_D14 + final_df_list_D7
final_df_pred = pd.concat(final_df_list)

# try:
#  pandas_gbq.to_gbq(
#     final_df_pred, 'project_656_prod_dataset.pltv_prediction', project_id=project_id, if_exists='replace')
# except:
#     filename="C:\Logs_from_models\log"+ str(now)[:10] +".txt"
#     f = open(filename, "w+")
#     f.write("problem with uploading data to bigquey")
#     f.close()

#####################write to bigquery table ###############################
job_config = bigquery.LoadJobConfig(
   write_disposition="WRITE_TRUNCATE"
)
job = client.load_table_from_dataframe(
    final_df_pred.reset_index(), 'cooladata-c-ilyon.project_121656_prod_dataset.pltv_prediction',job_config=job_config
)
# Wait for the load job to complete.
job.result()
#now = datetime.datetime.now()
#final_df_pred.to_csv(r'C:/Users/idan_w/PycharmProjects/LTV/Outputs/final_df_pred'+str(now)[:10]+".csv")









# upper_min_cluster_bound = grouped_df_test_with_pred.quantile(0.95)[' Net_IAP_7d '][min_cluster]
# lower_max_cluster_bound = grouped_df_test_with_pred.quantile(0.05)[' Net_IAP_7d '][max_cluster]
#
# if(lower_max_cluster_bound<upper_min_cluster_bound):
#     if((upper_min_cluster_bound-lower_max_cluster_bound)>=4):
#         f = open("C:\Logs_from_models\log_"+str(now)[:16]+".txt", "w+")
#         f.write("upper_min_cluster_bound-lower_max_cluster_bound: %s",(upper_min_cluster_bound-lower_max_cluster_bound))
#         sys.exit('exit because of upper_min_cluster_bound-lower_max_cluster_bound is greater than 4')
#     else:

#df_test_with_pred_D7[df_test_with_pred_D7['install_date'].apply(lambda x: True if(now.date()-x).days>=13 else False)]







# from sklearn.metrics import davies_bouldin_score
# from matplotlib import pyplot as plt
# wcss = []
# db_index = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=600, n_init=10, random_state=0)
#     kmeans.fit(df_test_std_D7)
#     wcss.append(kmeans.inertia_)
#     if(i>1):
#      labels = kmeans.labels_
#      db_index.append(davies_bouldin_score(df_test_std_D7, labels))
#
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()
#
# plt.plot(range(1, 10), db_index)
# plt.title('db index Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('DB INDEX')
# plt.show()



