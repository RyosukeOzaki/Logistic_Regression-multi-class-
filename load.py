import numpy as np
from sklearn import datasets

def data_load():
    mnist = datasets.fetch_mldata('MNIST original')
    dataset_X,dataset_Y = mnist.data,mnist.target
    dataset_X = dataset_X/255.
    one_hot_y = np.zeros((len(dataset_Y),10))
    for i,id in enumerate(dataset_Y):
        one_hot_y[i][int(id)] = 1
    n = len(dataset_X)
    s = np.random.permutation(n)
    train_size = int(n*0.8)
    test_size = int(n*0.1)
    train_X,dev_X,test_X = dataset_X[s[:train_size]],dataset_X[s[train_size:train_size+test_size]],dataset_X[s[train_size+test_size:]]
    train_Y,dev_Y,test_Y = one_hot_y[s[:train_size]],one_hot_y[s[train_size:train_size+test_size]],one_hot_y[s[train_size+test_size:]]
    return train_X,train_Y,dev_X,dev_Y,test_X,test_Y
