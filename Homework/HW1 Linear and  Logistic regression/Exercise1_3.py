# -*- coding: utf-8 -*-
"""
This module applies the gradient descent algorithm to dataset E2006.train.bz2,
aimed at acquiring coefficient vector omega. Then use this omega on dataset
E2006.test.bz2 and calculate mse.
MSE=1/N_{test} *sum{(x_{i}^{T}*w-y_{1})^2}
@date: April/22/2018
@auther: Mingyi Xue
"""
import numpy as np
import datetime
from sklearn import datasets
from scipy import sparse
import matplotlib.pyplot as plt


# # # # # # # # # # # # # # # # # # # #
# initialize data set
filename = r'C:\Users\薛明怡\Desktop\2018 spring quarter\STA 141C\Homework\HW1\E2006.train.bz2'
print("Begin to load data from:", filename)
start = datetime.datetime.now()
X, Y = datasets.load_svmlight_file(filename)
# X is a training data set, y is the values of each x
finish = datetime.datetime.now()
print("Dataset load finished, time cost:%s s" % str((finish-start).seconds))
X = X[:, :-2]
print("shape of training X:", X.shape, "type of X:", type(X))
Y = sparse.csr_matrix(Y).T
print("shape of training Y:", Y.shape, "type of Y:", type(Y))
# transpose Y to a column vector

filename = r'C:\Users\薛明怡\Desktop\2018 spring quarter\STA 141C\Homework\HW1\E2006.test.bz2'
print("Begin to load data from:", filename)
start = datetime.datetime.now()
test_X, test_Y = datasets.load_svmlight_file(filename)
finish = datetime.datetime.now()
print("Dataset load finished, time cost:%s s" % str((finish-start).seconds))
test_Y = sparse.csr_matrix(test_Y).T
print("shape of test X:", test_X.shape, "type of test X:", type(test_X))
print("shape of test Y:", test_Y.shape, "type of test Y:", type(test_Y))

# # # # # # # # # # # # # # # # # # # #
# initialize parameters
N, D = X.shape
# number of training data
n, d = test_X.shape
Lamda = 1
iter = 2000
# number of iteration
epsilon = 0.001
# converge condition

# # # # # # # # # # # # # # # # # # # #
# training
for i in range(-4, -1):
    learning_rate = 1.0*10**i
    print("learning rate equals ", str(learning_rate))
    # step size
    Yhat = []
    costs = []
    # # # # # # # # # # # # # # # # # # # #
    # initialize vector of omega
    w = np.random.randn(D)
    w = sparse.csr_matrix(np.matrix(w)).T
    print("shape of w:", w.shape, "type of w:", type(w))
    result0 = X.T@(X@w - Y) * 2 / N + Lamda * w
    result0 = np.sqrt((result0.T@result0)[0, 0])
    print("r0:", result0)
    # # # # # # # # # # # # # # # # # # # #
    for t in range(iter):
        # update w
        Yhat = X@w
        delta = Yhat - Y
        result = X.T@delta * 2 / N + Lamda * w
        if np.sqrt(result.T.dot(result)[0, 0]) < epsilon * result0:
            print("Converge!")
            break
        # gradient of loss function
        w = w - learning_rate * result
        # update coefficient vector w
        cost = (delta.T@delta)[0, 0] / N + Lamda/2 * (w.T@w)[0, 0]
        # print(cost)
        # find and store the cost
        costs.append(cost)
    # # # # # # # # # # # # # # # # # # # #
    # print out optimized w vector
    # print("final value of w:", w)
    print("final value of cost function:", costs[-1])

    # # # # # # # # # # # # # # # # # # # #
    # calculate mse for test set
    delta = test_X@w - test_Y
    mse = (delta.T@delta)[0, 0] / n
    print("mse:%f" % mse)

    # # # # # # # # # # # # # # # # # # # #
    # draw plots
    fig = plt.figure(figsize=(8, 16))
    # plot the costs
    plt.subplot(2, 1, 1)
    plt.plot(costs, label='reduction in loss function')
    plt.xlabel('number of iteration')
    plt.ylabel('value of loss function')
    plt.legend()
    # plot prediction vs target
    plt.subplot(2, 1, 2)
    a = np.array(sparse.csr_matrix.todense(Y).T)
    b = np.array(sparse.csr_matrix.todense(Yhat).T)
    plt.plot(a, b, 'oc', alpha=0.5, label='standardized target vs. prediction')
    plt.xlabel('target of y')
    plt.ylabel('prediction of y')
    target_file = r'D:\wkspacePY\STA 141C\HW1\Exercise1_3' + str(i) + '.png'
    plt.savefig(target_file)
# result of mean of omega
#[[7.50218239e-05]
# [9.03051401e-05]
# [1.84594515e-04]
# [1.61129326e-04]
# [1.52256153e-04]
# [1.28981781e-04]
# [9.91493428e-05]
# [1.38013398e-04]
# [1.21603850e-04]
# [4.23579909e-06]
# [1.49499142e-04]
# [2.37784495e-04]]
