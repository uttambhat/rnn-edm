import numpy as np

def prepare(time_series_data,X_column_list,y_column_list,number_of_delays,test_fraction):
    time_series_length = time_series_data.shape[0]
    train_length = np.int((1.-test_fraction)*time_series_length)-(number_of_delays+1)
    test_length = time_series_length-train_length-2*(number_of_delays+1)
    
    if len(time_series_data.shape)==1:
        time_series_data = np.atleast_2d(time_series_data).T
    
    X_train = []
    for i in range(number_of_delays):
        X_train.append(time_series_data[i:train_length+i,X_column_list])
    
    X_train = (np.atleast_3d(np.swapaxes(np.array(X_train),1,0)))
    y_train = (np.atleast_2d(time_series_data[number_of_delays:train_length+number_of_delays,y_column_list]))
    
    X_test = []
    for i in range(number_of_delays):
        X_test.append(time_series_data[train_length+number_of_delays+1+i:train_length+number_of_delays+1+i+test_length,X_column_list])
    
    X_test = (np.atleast_3d(np.swapaxes(np.array(X_test),1,0)))
    y_test = (np.atleast_2d(time_series_data[train_length+2*number_of_delays+1:train_length+2*number_of_delays+1+test_length,y_column_list]))
    
    return X_train,y_train,X_test,y_test

