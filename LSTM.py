from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM
from numpy import array
from numpy import hstack
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
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

#constans list:
COLUMNS_LSTM = ['Transc_7d','Net_IAP_7d','Transc_14d','Net_IAP_14d']
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
 SELECT *  FROM `cooladata-c-ilyon.project_121656_prod_dataset.pltv_training_data_reg_model`
  """
#################################################################

df_train_lstm = client.query(sql, project=project_id).to_dataframe()  # get train data

n_features = len(COLUMNS_LSTM)
n_steps= 2


model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')







X ,y = list(),list()
for i in range(len(df_train_lstm)):
    X.append(np.array([[df_train_lstm['Transc_7d'][i], df_train_lstm['Net_IAP_7d'][i]],
                       [df_train_lstm['Transc_14d'][i], df_train_lstm['Net_IAP_14d'][i]]]))
    y.append(df_train_lstm['Net_IAP_30d'][i])



# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
a = array([  [df_train_lstm['Transc_7d'][i],df_train_lstm['Net_IAP_7d'][i]]   for i in range(len(df_train_lstm))  ])

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))


