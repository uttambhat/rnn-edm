import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn.metrics as metrics

import rnn as rnn
import time_series_to_ml_edm as data_prep

########## Get data ################
time_series_data = np.loadtxt("time_series.txt")

########## Normalize time-series values #############
mean = np.mean(time_series_data)
stdev = np.sqrt(np.var(time_series_data))
time_series_data_normalized = (time_series_data-mean)/stdev

################## Data preparation ############################

X_column_list = [0]
y_column_list = [0]
number_of_delays = 2
test_fraction = 0.5

X_train,y_train,X_test,y_test = data_prep.prepare(time_series_data_normalized,X_column_list,y_column_list,number_of_delays,test_fraction)

########### Model RNN using rnn class ###############
rnn1 = rnn.rnn()
rnn1.initialize(X_train, y_train,hidden_units_1=3,hidden_units_2=3)
rnn1.get_parameters()

initial_theta=np.asarray(list(rnn1.parameters.values()))
(loss_vector,loss_val_vector)=rnn1.train(rnn1.X,rnn1.y,initial_theta,train_validation_split=0.7,epochs=10000)

y_pred=rnn1.predict(X_test)
rnn1.print_dashed_line()
print("Test error:")
print(metrics.mean_squared_error(y_test[:,0],y_pred[:,0]))
rnn1.print_dashed_line()

############ Plotting loss function, attractor and time-series predictions ###############
print("Plotting training and validation losses...")
plt.plot(loss_vector,label="Training")
plt.plot(loss_val_vector,label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation losses")
plt.show()
rnn1.print_dashed_line()

print("Plotting 2D time-delay coordinates")
plt.scatter(X_test[:,0,0],y_test[:,0],label="Data")
plt.scatter(X_test[:,0,0],y_pred[:,0],label="Prediction")
plt.xlabel(r'$x_{t-1}$')
plt.ylabel(r'$x_{t}$')
plt.legend()
plt.title("Time-delay coordinates (2D)")
plt.show()
rnn1.print_dashed_line()

print("Plotting 3D time-delay coordinates")
ax=plt.axes(projection='3d')
ax.scatter3D(X_test[:,0,0],X_test[:,1,0],y_test[:,0],label="Data")
ax.scatter3D(X_test[:,0,0],X_test[:,1,0],y_pred[:,0],label="Prediction")
ax.set_xlabel(r'$x_{t-2}$')
ax.set_ylabel(r'$x_{t-1}$')
ax.set_zlabel(r'$x_{t}$')
ax.legend()
ax.set_title("Time-delay coordinates (3D)")
plt.show()
rnn1.print_dashed_line()

print("Comparing test data and prediction...")
plt.plot(y_test[:100,0],label="Data")
plt.plot(y_pred[:100,0],label="Prediction")
plt.xlabel("time")
plt.ylabel("Value")
plt.legend()
plt.title("Data vs. Prediction")
plt.show()


