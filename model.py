import os
import numpy as np

class Softmax_Regression(object):
    def __init__(self, n_in, n_out, lr, minibatch_size):
        self.w = np.zeros((n_out,n_in))
        self.b = np.zeros(n_out)
        self.learn_rate = lr
        self.minibatch_size = minibatch_size

    def linear(self,x):
        return np.dot(x,self.w.T)+self.b

    def softmax(self,x):
        return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

    def forward(self,x):
        pre_y = self.softmax(self.linear(x))
        return pre_y

    def backward(self,input_x,pre_y,label_y):
        temp = pre_y - label_y
        delta_w = np.dot(temp.T,input_x)
        delta_b = np.sum(temp,axis=0)/len(label_y)
        return delta_w,delta_b

    def update(self,delta_w,delta_b):
        self.w -= delta_w*self.learn_rate
        self.b -= delta_b*self.learn_rate

    def cross_entropy_function(self,pre_y,label_y):
        e=0.00001
        cross_entropy_loss = -np.sum(label_y*np.log(pre_y+e))/len(label_y)
        return cross_entropy_loss

    def accuracy(self,pre_y,label_y):
        pre_y = np.argmax(pre_y,axis=1)
        y = np.argmax(label_y,axis=1)
        return np.sum(pre_y==y)/len(label_y)
