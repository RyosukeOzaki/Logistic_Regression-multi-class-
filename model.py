import os
import numpy as np

class Softmax_Regression(object):
    def __init__(self, n_in, n_out, lr, minibatch_size):
        self.W = np.zeros((n_out,n_in))
        self.b = np.zeros(n_out)
        self.learn_rate = lr
        self.minibatch_size = minibatch_size

    def linear(self,input_x):
        return np.dot(input_x,self.W.T)+self.b

    def softmax(self,input_x):
        return np.exp(input_x)/np.sum(np.exp(input_x))

    def forward(self,input_x):
        pre_y = self.softmax(self.linear(input_x))
        return np.argmax(pre_y,axis=1)

    def backward(self,input_x,label_y):
        pre_y = self.softmax(self.linear(input_x))
        temp = (pre_y - label_y)/len(label_y)
        delta_w = np.dot(temp.T,input_x)
        delta_b = np.sum(temp,axis=0)
        return delta_w,delta_b

    def update(self,input_x,label_y):
        delta_w,delta_b = self.backward(input_x,label_y)
        self.W -= delta_w*self.learn_rate
        self.b -= delta_b*self.learn_rate


    def cross_entropy_function(self,input_x,label_y):
        e=0.00001
        pre_y = self.softmax(self.linear(input_x))
        cross_entropy_loss = -np.sum(label_y*np.log(pre_y+e))/len(label_y)
        return cross_entropy_loss

    def accuracy(self,input_x,label_y):
        pre_y = self.forward(input_x)
        y = np.argmax(label_y,axis=1)
        return np.sum(pre_y==y)/len(label_y)
