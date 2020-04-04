######### change path according to where you store homemade library files #########
import sys, os
sys.path.append(os.path.abspath("../..")+"/python_homemade_commons")

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn.metrics as metrics

import rnn as rnn
import ml_data_prep.time_series_to_ml_edm as time_series_to_ml_edm

########## Get data ################
time_series_data = np.loadtxt("../python_time_series_generators/time_series_data/time_series_02.txt")

########## Normalize time-series values #############
mean = np.mean(time_series_data)
stdev = np.sqrt(np.var(time_series_data))
time_series_data_normalized = (time_series_data-mean)/stdev
#time_series_data_normalized = time_series_data

################## Data preparation ############################

X_column_list = [0,1]
y_column_list = [0]
number_of_delays = 1
test_fraction = 0.5

X_train,y_train,X_test,y_test = time_series_to_ml_edm.prepare(time_series_data_normalized,X_column_list,y_column_list,number_of_delays,test_fraction)
########## uncomment to linearize Ricker time-series ##################
#X_train=X_train[1:]
#y_train = np.log(y_train[1:]/y_train[:-1])
#X_test = X_test[1:]
#y_test = np.log(y_test[1:]/y_test[:-1])

########### Model RNN using rnn class ###############
rnn1 = rnn.rnn()
rnn1.fit(X_train, y_train,hidden_units=10)
rnn1.get_parameters()

theta=np.array([rnn1.Wxh, rnn1.Whh, rnn1.bh, rnn1.Whx, rnn1.bx])
(loss_vector,loss_val_vector)=rnn1.train(rnn1.X,rnn1.y,theta,train_validation_split=0.7,epochs=20000,stateful=False)


y_pred=rnn1.predict(X_test)
print(metrics.mean_squared_error(y_test[:,0],y_pred[:,0]))





plt.plot(loss_vector)
plt.plot(loss_val_vector)
plt.show()


plt.scatter(X_test[:,0,0],y_test[:,0])
plt.scatter(X_test[:,0,0],y_pred[:,0])
plt.show()

ax=plt.axes(projection='3d')
ax.scatter3D(X_test[:,0,0],X_test[:,0,1],y_test[:,0])
ax.scatter3D(X_test[:,0,0],X_test[:,0,1],y_pred[:,0])
plt.show()

plt.plot(y_test[:100,0])
plt.plot(y_pred[:100,0])
plt.show()


