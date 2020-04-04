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
time_series_data = np.loadtxt("../python_time_series_generators/time_series_data/time_series_01.txt")

########## Normalize time-series values #############
mean = np.mean(time_series_data)
stdev = np.sqrt(np.var(time_series_data))
time_series_data_normalized = (time_series_data-mean)/stdev


################## Data preparation ############################
test_fraction = 0.5

########### Loop over number of delays and hidden units
errors2 = []
max_delays = 6
max_hidden_layer_size = 10
delays_range = range(1,max_delays+1)
hidden_layer_size_range = range(1,max_hidden_layer_size+1)
for number_of_delays in delays_range:
    for hidden_layer_size in hidden_layer_size_range:
        X_train,y_train,X_test,y_test = time_series_to_ml_edm.prepare(time_series_data_normalized,number_of_delays,test_fraction)
        
        
        ########### Model RNN using rnn class ###############
        rnn1 = rnn.rnn()
        rnn1.fit(X_train, y_train,hidden_units=hidden_layer_size)
        rnn1.get_parameters()
        
        theta=np.array([rnn1.Wxh, rnn1.Whh, rnn1.bh, rnn1.Whx, rnn1.bx])
        (loss_vector,loss_val_vector)=rnn1.train(rnn1.X,rnn1.y,theta,train_validation_split=0.7,epochs=20000,stateful=False)
        y_pred=rnn1.predict(X_test)
        errors2.append(metrics.mean_squared_error(y_test[:,0],y_pred[:,0]))


errors = np.array(errors).reshape(max_delays,max_hidden_layer_size)

fig,ax = plt.subplots()
im = ax.imshow(errors2)
ax.set_xticks(np.arange(len(hidden_layer_size_range)))
ax.set_yticks(np.arange(len(delays_range)))
ax.set_xticklabels(hidden_layer_size_range)
ax.set_yticklabels(delays_range)
ax.set_xlabel("Hidden layer size")
ax.set_ylabel("Number of delays")
for i in range(len(delays_range)):
    for j in range(len(hidden_layer_size_range)):
        text = ax.text(j, i, format(errors2[i,j],'.2g'), ha="center", va="center", color="w")

ax.set_title("MSE")
fig.tight_layout()
plt.show()



heatmap=plt.pcolor(errors)
plt.colorbar(heatmap)
plt.show()


plt.plot(loss_vector)
plt.plot(loss_val_vector)
plt.show()

plt.scatter(X_test[:,0,0],y_test[:,0])
plt.scatter(X_test[:,0,0],y_pred[:,0])
plt.show()

ax=plt.axes(projection='3d')
ax.scatter3D(X_test[:,0,0],X_test[:,1,0],y_test[:,0])
ax.scatter3D(X_test[:,0,0],X_test[:,1,0],y_pred[:,0])
plt.show()

plt.plot(y_test[:100,0])
plt.plot(y_pred[:100,0])
plt.show()


