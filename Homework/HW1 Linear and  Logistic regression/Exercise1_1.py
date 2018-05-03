# -*- coding: utf-8 -*-
"""
This module implements a simple gradient descent algorithm for machine learning on dataset cpusmall.
Loss Function(w) =  argmin{1/n sum(x_{i}^{T} * w - y_{i})^2 + lamda/2 * w^{T} * w}
@date: April/22/2018
@auther: Mingyi Xue
"""


import numpy as np
import matplotlib.pyplot as plt
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
print("Dataset load finished, time cost:%s s" % str((finish-start).seconds))
X = np.matrix(sparse.csr_matrix.todense(X))
Y = np.matrix(Y).T
print("shape of X:", X.shape, "type of X:", type(X))
print("shape of Y:", Y.shape, "type of Y:", type(Y))
# print X so you know what it looks like
print("head of X:", X[:3, :])
# standardized X
X = sk.preprocessing.normalize(X, axis=0)
Y = sk.preprocessing.normalize(Y, axis=0)

# # # # # # # # # # # # # # # # # # # #
# initialize parameters
N = len(X)
# number of data
D = len(X[0, :])
# dimension of data
Lamda = 1
# ridge regression parameter
iter = 5000
# number of iteration
epsilon = 0.001
# converge condition


# # # # # # # # # # # # # # # # # # # #
# iteration
for i in range(-7, -1):
    learning_rate = 1.0*10**i
    # eta/learning rate
    Yhat = []
    # estimated value of y
    costs = []
    # keep track of squared error cost

    # # # # # # # # # # # # # # # # # # # #
    # initialize coefficients
    w = np.zeros(D)
    w = np.matrix(w).T
    print("shape of w:", w.shape, "type of w:", type(w))
    print("value of initial omega:", w)
    result0 = X.transpose().dot(X.dot(w) - Y) * 2 / N + Lamda * w
    result0 = np.sqrt(result0.T.dot(result0)[0, 0])
    # # # # # # # # # # # # # # # # # # # #
    print("learning rate equals ", str(learning_rate))
    for t in range(iter):
        # update w
        Yhat = X.dot(w)
        delta = Yhat - Y
        result = X.transpose().dot(delta) * 2 / N + Lamda * w
        # gradient of loss function
        r = np.sqrt(result.T.dot(result)[0, 0])
        # print(r)
        if r < epsilon * result0:
            print("Converge!")
            break
        w = w - learning_rate * result
        # update coefficient vector w
        cost = delta.T.dot(delta)[0, 0] / N + Lamda/2 * w.T.dot(w)[0, 0]
        # find and store the cost
        costs.append(cost)

    # # # # # # # # # # # # # # # # # # # #
    # print out optimized w vector
    print("final value of w:", w)
    print("final value of loss function:", costs[-1])
    # # # # # # # # # # # # # # # # # # # #
    # draw plots
    fig = plt.figure(figsize=(8, 16))
    # plot the costs
    plt.subplot(2, 1, 1)
    # plt.axis([0, iter+1, 0, 10**10])
    plt.plot(costs, label='reduction in loss function')
    plt.xlabel('number of iteration')
    plt.ylabel('value of loss function')
    plt.legend()
    # plot prediction vs target
    plt.subplot(2, 1, 2)
    # plt.axis([0, 100])
    plt.plot(Y, Yhat, 'oc', alpha=0.5, label='standardized target vs. prediction')
    plt.xlabel('target of y')
    plt.ylabel('prediction of y')
    # plt.plot(Y, '.k', alpha=0.5, label='target')
    plt.legend()
    target_file = r'D:\wkspacePY\STA 141C\HW1\Exercise1_1'+str(i)+'.png'
    plt.savefig(target_file)