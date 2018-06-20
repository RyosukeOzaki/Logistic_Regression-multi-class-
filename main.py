import os,sys
from load import data_load
from batch import batcher
from model import *

def evaluate(SR,input_x,label_y):
    Loss = SR.cross_entropy_function(input_x,label_y)
    Accuracy = SR.accuracy(input_x,label_y)
    return Loss,Accuracy

if __name__=='__main__':
    train_x,train_y,dev_x,dev_y,test_x,test_y = data_load()

    minibatch_size = 100
    epoch = 100
    n_in =  len(train_x[0])
    n_out = len(train_y[0])
    learn_rate = 0.001
    #train
    SR = Softmax_Regression(n_in,n_out,learn_rate,minibatch_size)
    for ep in range(epoch):
        for input_x,label_y in batcher(train_x,train_y,minibatch_size):
            Loss,Accuracy = evaluate(SR,input_x,label_y)
            SR.update(input_x,label_y)
            #print('Train | Epoch{0} | Data size{1} | Loss{2:.2f} | Accuracy{3:.2f}%'.format(ep+1,len(label_y),Loss,Accuracy*100))
        Loss,Accuracy = evaluate(SR,dev_x,dev_y)
        print('Dev | Epoch{0} | Data size{1} | Loss{2:.2f} | Accuracy{3:.2f}%'.format(ep+1,len(dev_y),Loss,Accuracy*100))
    #test
    Loss,Accuracy = evaluate(SR,input_x,label_y)
    print('test | Data size{0} | Loss{1:.2f} | Accuracy{2:.2f}%'.format(len(dev_y),Loss,Accuracy*100))
