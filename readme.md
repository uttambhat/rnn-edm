# Recurrent Neural Network for Empirical Dynamical Modeling

Fits a Recurrent Neural Network on delay-coordinate vectors from the input time-series.

### Usage

1. Download this folder. 
2. Change "time-series.txt" in line 12 of main.py to the path to the data file
3. Specify columns in data file to be used as X (line 21) and y (line 22)
4. Specify the number of delays/lags (line 23)
5. Specify the fraction of data to be used as test data (line 24)
6. Specify the number hidden neurons (line 30). There are two possible RNN architectures. To fit a single 'hidden function' RNN, use None for hidden_units_2. Specifying integers for both fits a double 'hidden function' RNN with different weights & biases for F and G (refer paper)
7. Specify the train-validation split (i.e., fraction of X_train to be used for training) and the number of epochs (line 34).
8. Run main.py
