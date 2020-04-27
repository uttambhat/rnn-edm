####################################################
## Ws and bs as OrderedDictionaries to avoid #######
## hand-ordering ###################################
####################################################
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict

class rnn:
    def __init__(self,learning_rate=1.e-3,optimizer="rmsprop", n_restarts_optimizer=100,epsilon=1.e-5):
        self.learning_rate = learning_rate
        self.optimizer=optimizer
        self.n_restarts_optimizer=n_restarts_optimizer
        self.epsilon=epsilon
    
    def fit(self,X_train,y_train,hidden_units_1=2,hidden_units_2=None):
        self.X=np.transpose(X_train, (2,1,0))
        self.y=np.transpose(y_train, (1,0))

        ######### Model parameters ###########
        self.x_size=self.X.shape[0]
        self.y_size=self.y.shape[0]
        self.num_delays = self.X.shape[1]
        self.batch_size = self.X.shape[2]
        if(hidden_units_2==None):
            self.h_size = hidden_units_1
        else:
            self.f_size = hidden_units_1
            self.g_size = hidden_units_2
            

        #weights and biases
        self.parameters = OrderedDict()
        if(hidden_units_2==None):
            self.parameters['Wxh'] = np.random.randn(self.h_size, self.x_size)*0.01 # h to x
            self.parameters['Whh'] = np.random.randn(self.h_size, self.h_size)*0.01 # h to h
            self.parameters['Whx'] = np.random.randn(self.y_size, self.h_size)*0.01 # h to y
            self.parameters['bh'] = np.zeros((self.h_size, 1)) # x bias
            self.parameters['bx'] = np.zeros((self.y_size, 1)) # y bias
        else:
            self.parameters['Wxg'] = np.random.randn(self.g_size, self.x_size)*0.01 # h to x
            self.parameters['Wgg'] = np.random.randn(self.g_size, self.g_size)*0.01 # h to h
            self.parameters['Wxf'] = np.random.randn(self.f_size, self.x_size)*0.01 # h to x
            self.parameters['Wgf'] = np.random.randn(self.f_size, self.g_size)*0.01 # h to h
            self.parameters['Wfx'] = np.random.randn(self.y_size, self.f_size)*0.01 # h to y
            self.parameters['bg'] = np.zeros((self.g_size, 1)) # x bias
            self.parameters['bf'] = np.zeros((self.f_size, 1)) # x bias
            self.parameters['bx'] = np.zeros((self.y_size, 1)) # y bias
        
        self.total_parameters = 0
        for key in self.parameters.keys():
            self.total_parameters += self.parameters[key].size
        
    
    def print_dashed_line(self):
        print("--------------------------------------------")
    
    def get_parameters(self):
        self.print_dashed_line()
        print(" Parameters and dimensions ")
        self.print_dashed_line()
        for key in self.parameters.keys():
            print(str(key)+": (",self.parameters[key].shape[0],"x",self.parameters[key].shape[1],") - ",self.parameters[key].size," parameters")
        
        print("Total parameters = ",self.total_parameters)
        self.print_dashed_line()
        print(" Values ")
        self.print_dashed_line()
        for key in self.parameters.keys():
            print(str(key)+": ")
            self.print_dashed_line()
            print(self.parameters[key])
            self.print_dashed_line()
        
    def forward_prop(self,X_train,theta,stateful=False):
        parameters = OrderedDict(zip(self.parameters.keys(),theta))
        dparameters = OrderedDict()
        for key in parameters.keys():
            dparameters[key] = np.zeros_like(parameters[key])
        
        X=X_train
        
        num_delays=X_train.shape[1]
        batch_size=X_train.shape[2]
        
        if(len(self.parameters)==5):
            h = {}
            h[-1] = np.zeros_like(np.dot(parameters['Wxh'],X[:,0,:]) + parameters['bh'])
            #Calculate loss
            for i in range(num_delays):
                h[i] = np.tanh( np.dot(parameters['Wxh'],X[:,i,:]) + np.dot(parameters['Whh'],h[i-1]) + parameters['bh'])
            
            y_pred = np.dot(parameters['Whx'],h[num_delays-1])+parameters['bx']
        else:
            g = {}
            g[-1] = np.zeros_like(np.dot(parameters['Wxg'],X[:,0,:]) + parameters['bg'])
            
            #Calculate loss
            num_delays=X_train.shape[1]
            for i in range(num_delays-1):
                g[i] = np.tanh( np.dot(parameters['Wxg'],X[:,i,:]) + np.dot(parameters['Wgg'],g[i-1]) + parameters['bg'])
            
            f = np.tanh( np.dot(parameters['Wxf'],X[:,num_delays-1,:]) + np.dot(parameters['Wgf'],g[num_delays-2]) + parameters['bf'])
            y_pred = np.dot(parameters['Wfx'],f)+parameters['bx']
            h=(f,g)
        
        return y_pred,h

    def loss_function(self,X_train,y_train,theta,stateful=False):
        batch_size=X_train.shape[2]
        y_pred,h = self.forward_prop(X_train,theta,stateful)
        dy = 2*(y_train-y_pred)
        loss = np.sum(np.power(dy/2.,2))/batch_size
        return loss,dy,h
    
    def backprop(self,X_train,y_train,theta,stateful=False):
        loss,dy,h = self.loss_function(X_train,y_train,theta,stateful)
        #unpack theta
        parameters = OrderedDict(zip(self.parameters.keys(),theta))
        dparameters = OrderedDict()
        for key in parameters.keys():
            dparameters[key] = np.zeros_like(parameters[key])
        
        batch_size=X_train.shape[2]
        num_delays=X_train.shape[1]
        X=X_train
        y=y_train
        
        if(len(self.parameters)==5):
            #Calculate gradients dWxh, dbh, dWhx, dbx
            dparameters['Whx'] += np.dot(dy, h[num_delays-1].T)
            dparameters['bx'] += np.dot(dy, np.ones((dy.shape[1],dparameters['bx'].shape[1])))
            
            dh = np.dot(parameters['Whx'].T, dy) # backprop into h
            for i in reversed(range(num_delays)):
                dz = (1. - h[i]*h[i]) * dh # backprop through tanh nonlinearity
                dparameters['Wxh'] += np.dot(dz, X[:,i,:].T)
                dparameters['Whh'] += np.dot(dz, h[i-1].T)
                dparameters['bh'] += np.dot(dz, np.ones((dz.shape[1],dparameters['bh'].shape[1])))
                dh = np.dot(parameters['Whh'].T, dz)
            
        else:
            (f,g)=h
            #Calculate gradients dWxf, dbf, dWfx, dbx, dWgf, dWxg, dWgg, dbg
            dparameters['Wfx'] += np.dot(dy, f.T)
            dparameters['bx'] += np.dot(dy, np.ones((dy.shape[1],dparameters['bx'].shape[1])))
            
            ### TO BE ADAPTED FOR FG BELOW ####
            df = np.dot(parameters['Wfx'].T, dy) # backprop into f
            dz = (1. - f*f) * df # backprop through tanh nonlinearity
            dparameters['Wxf'] += np.dot(dz, X[:,num_delays-1,:].T)
            dparameters['Wgf'] += np.dot(dz, g[num_delays-2].T)
            dparameters['bf'] += np.dot(dz, np.ones((dz.shape[1],dparameters['bf'].shape[1])))
            dg = np.dot(parameters['Wgf'].T, dz)
            for i in reversed(range(num_delays-1)):
                dz = (1. - g[i]*g[i]) * dg # backprop through tanh nonlinearity
                dparameters['Wxg'] += np.dot(dz, X[:,i,:].T)
                dparameters['Wgg'] += np.dot(dz, g[i-1].T)
                dparameters['bg'] += np.dot(dz, np.ones((dz.shape[1],dparameters['bg'].shape[1])))
                dg = np.dot(parameters['Wgg'].T, dz)
            
        
        return loss,np.array(list(dparameters.values()))/batch_size
    
    def gradient_check(self,X_train,y_train,theta,stateful,epsilon):
        loss,grad = self.backprop(X_train,y_train,theta,stateful)
        numerical_grad=[]
        for i in range(grad.size):
            numerical_grad.append(np.copy(grad[i]))
        
        numerical_grad=np.array(numerical_grad)
        difference=0.
        for i in range(theta.size):
            for j in range(theta[i].shape[0]):
                for k in range(theta[i].shape[1]):
                    theta_plus, theta_minus = [], []
                    for l in range(theta.size):
                        theta_plus.append(np.copy(theta[l]))
                        theta_minus.append(np.copy(theta[l]))
                    
                    theta_plus=np.array(theta_plus)
                    theta_minus=np.array(theta_minus)
                    theta_plus[i][j,k] = theta[i][j,k]+epsilon
                    theta_minus[i][j,k] = theta[i][j,k]-epsilon
                    loss_plus,_,_ = self.loss_function(X_train,y_train,theta_plus,stateful)
                    loss_minus,_,_ = self.loss_function(X_train,y_train,theta_minus,stateful)
                    numerical_grad[i][j,k] = (loss_plus-loss_minus)/(2.*epsilon)
                    difference += np.abs(numerical_grad[i][j,k]-grad[i][j,k])

        return difference,numerical_grad,grad
    
    def train(self,X_train,y_train,initial_theta,train_validation_split=0.7,epochs=10000,stateful=False):
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
            (loss,grad) = self.backprop(X,y,theta,stateful)
            (loss_val,_,_) = self.loss_function(X_val,y_val,theta,stateful)
            
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
        
        self.parameters = OrderedDict(zip(self.parameters.keys(),theta)) 
        return loss_vector, loss_val_vector
    
    def predict(self,X,stateful=False):
        X=np.transpose(X, (2,1,0))
        theta=np.asarray(list(self.parameters.values()))
        y_pred,_=self.forward_prop(X,theta,stateful)
        return y_pred.T

