# -*- coding: utf-8 -*-
"""
This module implements a simple cross validation algorithm for machine learning on dataset cpusmall.
MSE=1/N_{test} *sum{(x_{i}^{T}*w-y_{1})^2}
@date: April/22/2018
@auther: Mingyi Xue
"""
import numpy as np
import datetime
from sklearn import datasets
from scipy import sparse
import sklearn as sk

# # # # # # # # # # # # # # # # # # # #
# initialize data set
filename = r'C:\Users\薛明怡\Desktop\2018 spring quarter\STA 141C\Homework\HW1\cpusmall.txt'
print("Begin to load data from:", filename)
start = datetime.datetime.now()
X, Y = datasets.load_svmlight_file(filename)
# X is a training data set, y is the values of each x
finish = datetime.datetime.now()
print("Data set load finished, time cost:%s s" % str((finish-start).seconds))
print("convert X to dense matrix")
X = sparse.csr_matrix.todense(X)
# X = np.matrix(np.ones(N)).T.hstack(X)
# add a column of 1 to the start of matrix X
Y = np.matrix(Y).T
# transpose Y to a column vector
print("shape of X:", X.shape, "type of X:", type(X))
print("shape of Y:", Y.shape, "type of Y:", type(Y))
# print X so you know what it looks like
print("head of X:", X[:3, :])
# standardized X, Y
X = sk.preprocessing.normalize(X, axis=0)
Y = sk.preprocessing.normalize(Y, axis=0)

# # # # # # # # # # # # # # # # # # # #
# initialize parameters
N = len(X)
# number of data
D = len(X[0, :])
# dimension of data
learning_rate = 1.0*10**(-2)
Lamda = 1
iter = 1000
# number of iteration
folder = 5
# number of folders

# # # # # # # # # # # # # # # # # # # #
# initialize vector of omega
w = np.zeros(D)
w = np.matrix(w).T
print("shape of w:", w.shape, "type of w:", type(w))
print("value of initial omega:", w)

# # # # # # # # # # # # # # # # # # # #
# slice the data set X to test data and training data
start = 0
interval = int(np.ceil(N/folder))
print("zipping X and Y...")
data = np.hstack((Y, X))
print("shuffling data...")
np.random.shuffle(data)
end = start + interval

costs = []
ws = []
# list of mse
for i in range(folder):
    if start >= len(data):
        break
    test_data = []
    train_data = []
    if end < len(data):
        test_data = np.matrix(data[start:end, :])
        train_data = np.matrix(np.vstack((data[0:start, :], data[end:, :])))
    else:
        test_data = np.matrix(data[start:, :])
        train_data = np.matrix(data[0:start, :])
    start = end
    end = start + interval
    train_X = train_data[:, 1:]
    train_Y = train_data[:, 0]
    test_X = test_data[:, 1:]
    test_Y = test_data[:, 0]
    for t in range(iter):
        # update w
        Yhat = train_X.dot(w)
        delta = Yhat - train_Y
        result = train_X.transpose().dot(delta) * 2 / len(train_data) + Lamda * w
        w = w - learning_rate * result
        # find and store the cost
    delta = test_X.dot(w)-test_Y
    mse = delta.T.dot(delta)[0, 0] / len(test_X)
    print("result of %d th validation" % (i+1))
    print("value of final w:", w)
    print("mse:", str(mse))
    ws.append(w)
    costs.append(mse)


print("Mean Square Error:", np.mean(costs))
print("mean w:", np.mean(ws, axis=0))








