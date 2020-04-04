import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class rnn:
    def __init__(self,learning_rate=1.e-3,optimizer="rmsprop", n_restarts_optimizer=100,epsilon=1.e-5):
        self.learning_rate = learning_rate
        self.optimizer=optimizer
        self.n_restarts_optimizer=n_restarts_optimizer
        self.epsilon=epsilon
    
    def fit(self,X_train,y_train,hidden_units=2):
        self.X=np.transpose(X_train, (2,1,0))
        self.y=np.transpose(y_train, (1,0))

        ######### Model parameters ###########
        self.x_size=self.X.shape[0]
        self.y_size=self.y.shape[0]
        self.h_size = hidden_units
        self.num_delays = self.X.shape[1]
        self.batch_size = self.X.shape[2]

        #weights and biases
        self.Wxh = np.random.randn(self.h_size, self.x_size)*0.01 # h to x
        self.Whh = np.random.randn(self.h_size, self.h_size)*0.01 # h to h
        self.Whx = np.random.randn(self.y_size, self.h_size)*0.01 # h to y
        self.bh = np.zeros((self.h_size, 1)) # x bias
        self.bx = np.zeros((self.y_size, 1)) # y bias
        
    
    def print_dashed_line(self):
        print("--------------------------------------------")
    
    def get_parameters(self):
        self.print_dashed_line()
        print(" Parameters and dimensions ")
        self.print_dashed_line()
        print("Wxh: (",self.Wxh.shape[0],"x",self.Wxh.shape[1],") - ",self.Wxh.size," parameters")
        print("Whh: (",self.Whh.shape[0],"x",self.Whh.shape[1],") - ",self.Whh.size," parameters")
        print("bh: (",self.bh.shape[0],"x",self.bh.shape[1],") - ",self.bh.size," parameters")
        print("Whx: (",self.Whx.shape[0],"x",self.Whx.shape[1],") - ",self.Whx.size," parameters")
        print("Whh: (",self.Whh.shape[0],"x",self.Whh.shape[1],") - ",self.Whh.size," parameters")
        print("bx: (",self.bx.shape[0],"x",self.bx.shape[1],") - ",self.bx.size," parameters")
        print("Total parameters = ",self.Wxh.size+self.bh.size+self.Whx.size+self.Whh.size+self.bx.size)
        self.print_dashed_line()
        print(" Values ")
        self.print_dashed_line()
        print("Wxh: ")
        print(self.Wxh)
        self.print_dashed_line()
        print("Whh: ")
        print(self.Whh)
        self.print_dashed_line()
        print("bh: ")
        print(self.bh)
        self.print_dashed_line()
        print("Whx: ")
        print(self.Whx)
        self.print_dashed_line()
        print("bx: ")
        print(self.bx)
        self.print_dashed_line()

    def loss_function(self,X_train,y_train,theta,stateful):
        #unpack theta
        Wxh=theta[0]
        Whh=theta[1]
        bh=theta[2]
        Whx=theta[3]
        bx=theta[4]
        
        X=X_train
        y=y_train
        
        batch_size=X_train.shape[2]
        
        h = {}
        h[-1] = np.zeros_like(np.dot(Wxh,X[:,0,:]) + bh)
        
        #Calculate loss
        num_delays=X_train.shape[1]
        for i in range(num_delays):
            h[i] = np.tanh( np.dot(Wxh,X[:,i,:]) + np.dot(Whh,h[i-1]) + bh)
        
        y_pred = np.dot(Whx,h[num_delays-1])+bx
        dy = 2*(y-y_pred) #will be used later in grad-calculation
        loss = np.sum(np.power(dy/2.,2))
        
        #Calculate gradients dWxh, dbh, dWhx, dbx
        dWxh, dWhh, dWhx = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Whx)
        dbh, dbx = np.zeros_like(bh), np.zeros_like(bx)
        dhnext = np.zeros_like(bh)
        
        dWhx += np.dot(dy, h[num_delays-1].T)
        dbx += np.dot(dy, np.ones((dy.shape[1],dbx.shape[1])))
        
        dh = np.dot(Whx.T, dy) # backprop into h
        for i in reversed(range(num_delays)):
            dz = (1. - h[i]*h[i]) * dh # backprop through tanh nonlinearity
            dWxh += np.dot(dz, X[:,i,:].T)
            dWhh += np.dot(dz, h[i-1].T)
            dbh += np.dot(dz, np.ones((dz.shape[1],dbh.shape[1])))
            dh = np.dot(Whh.T, dz)
        
        return loss/batch_size,np.array([dWxh,dWhh,dbh,dWhx,dbx])/batch_size

#    def gradient_check(self,X_train,y_train,theta,stateful):
    
    def train(self,X_train,y_train,initial_theta,train_validation_split=0.7,epochs=10000,stateful=False):
        #theta=np.array([rnn1.Wxh, rnn1.Whh, rnn1.bh, rnn1.Whx, rnn1.bx])
        split_point=int(np.round(train_validation_split*X_train.shape[2]))
        X = X_train[:,:,:split_point]
        y = y_train[:,:split_point]
        X_val = X_train[:,:,split_point:]
        y_val = y_train[:,split_point:]
        
        theta=np.array(initial_theta)
        Eg2 = np.copy(theta*0)
        loss_vector=[]
        loss_val_vector=[]
        
        self.print_dashed_line()
        for i in range(epochs):
            (loss,grad) = self.loss_function(X,y,theta,stateful)
            (loss_val,grad_val) = self.loss_function(X_val,y_val,theta,stateful)
            
            loss_vector.append(loss)
            loss_val_vector.append(loss_val)
            if i==0:
                Eg2 = np.power(grad,2)
            else:
                Eg2 = 0.9*Eg2 + 0.1*np.power(grad,2)
            
            delta_theta = self.learning_rate*grad/np.power(Eg2 + 1.e-8,0.5)
            theta = theta + delta_theta
            if(i%(epochs/10)==0):
                print("Iteration: ",i," Training loss: ",loss," Validation loss: ",loss_val)
                #print("Theta: ",np.hstack(theta[0]),np.hstack(theta[1]),np.hstack(theta[2]),np.hstack(theta[3]),np.hstack(theta[4]))
                #print("Grad_theta: ",np.hstack(grad[0]),np.hstack(grad[1]),np.hstack(grad[2]),np.hstack(grad[3]),np.hstack(theta[4]))
                #print("Delta_theta: ",np.hstack(delta_theta[0]),np.hstack(delta_theta[1]),np.hstack(delta_theta[2]),np.hstack(delta_theta[3]),np.hstack(delta_theta[4]))
                #self.print_dashed_line()
         
        self.Wxh=theta[0]
        self.Whh=theta[1]
        self.bh=theta[2]
        self.Whx=theta[3]
        self.bx=theta[4]
        return loss_vector, loss_val_vector
    
    def predict(self,X):
        X=np.transpose(X, (2,1,0))
        h = {}
        h[-1] = np.zeros_like(np.dot(self.Wxh,X[:,0,:]) + self.bh)
        #Calculate loss
        num_delays=X.shape[1]
        for i in range(num_delays):
            h[i] = np.tanh( np.dot(self.Wxh,X[:,i,:]) + np.dot(self.Whh,h[i-1]) + self.bh)
        
        y_pred = np.dot(self.Whx,h[num_delays-1])+self.bx
        return y_pred.T

