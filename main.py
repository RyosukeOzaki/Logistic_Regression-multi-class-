import os,sys
import math
from load import data_load
from batch import batcher
from model import *

if __name__=='__main__':
    train_x,train_y,dev_x,dev_y,test_x,test_y = data_load()

    minibatch_size = 100
    epoch = 100
    n_in =  len(train_x[0])
    n_out = len(train_y[0])
    learn_rate = 0.01
    m = math.ceil(len(train_y)/minibatch_size)
    #train
    model = Softmax_Regression(n_in,n_out,learn_rate,minibatch_size)
    for ep in range(epoch):
        loss = 0
        accuracy = 0
        for input_x,label_y in batcher(train_x,train_y,minibatch_size):
            y = model.forward(input_x)
            accuracy += model.accuracy(y,label_y)
            loss += model.cross_entropy_function(y,label_y)
            delta_w,delta_b = model.backward(input_x,y,label_y)
            model.update(delta_w,delta_b)
        print('Train | Epoch{0} | Data size{1} | Loss{2:.2f} | Accuracy{3:.2f}%'.format(ep+1,len(train_y),loss/m,accuracy/m*100))
        if ep%10==0:
            y = model.forward(dev_x)
            accuracy = model.accuracy(y,dev_y)
            loss = model.cross_entropy_function(y,dev_y)
            print('Dev | Epoch{0} | Data size{1} | Loss{2:.2f} | Accuracy{3:.2f}%'.format(ep+1,len(dev_y),loss,accuracy*100))
    #test
    y = model.forward(test_x)
    accuracy = model.accuracy(y,test_y)
    loss = model.cross_entropy_function(y,test_y)
    print('Test | Data size{0} | Loss{1:.2f} | Accuracy{2:.2f}%'.format(len(test_y),loss,accuracy*100))
