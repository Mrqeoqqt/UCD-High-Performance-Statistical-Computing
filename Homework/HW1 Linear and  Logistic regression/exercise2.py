# -*- coding: utf-8 -*-
"""
This module applies logistic regression on dataset news20.binary.bz2.
gradient of loss function = (1/n)(X.T y)/(1+exp(w.T X.T y)) + w
@date: April/30/2018
@auther: Mingyi Xue
"""
import numpy as np
import datetime
from sklearn import datasets
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# # # # # # # # # # # # # # # # # # # #
# initialize data set
filename = r'C:\Users\薛明怡\Desktop\2018 spring quarter\STA 141C\Homework\HW1\news20.binary.bz2'
print("Begin to load data from:", filename)
start = datetime.datetime.now()
X, Y = datasets.load_svmlight_file(filename)
# X is a training data set, y is the values of each x
finish = datetime.datetime.now()
print("Dataset load finished, time cost:%s s" % str((finish-start).seconds))
print("shape of training X:", X.shape, "type of X:", type(X))
Y = sparse.csr_matrix(Y).T
print("shape of training Y:", Y.shape, "type of Y:", type(Y))
# transpose Y to a column vector
percent = 0.2
print("split dataset into %d %d" % (int(percent*100), int((1-percent)*100)))
X, test_X, Y, test_Y = train_test_split(X, Y, test_size=percent)


# # # # # # # # # # # # # # # # # # # #
# initialize parameters
N, D = X.shape
# number of training data
n, d = test_X.shape
Lamda = 1
iter = 5000
# number of iteration
epsilon = 0.001
# converge condition

# # # # # # # # # # # # # # # # # # # #
# iteration
learning_rate = 1.0 * 10 ** (-2)
print("learning rate equals ", str(learning_rate))
# step size
Yhat = []
costs = []


# # # # # # # # # # # # # # # # # # # #
# define a iterate function for matrix,
# apply this function to each element in the matrix
def f(x):
    return 1/(1+np.exp(x))


# # # # # # # # # # # # # # # # # # # #
# initialize vector of omega
w = np.random.randn(D)
w = sparse.csr_matrix(np.matrix(w)).T
print("shape of w:", w.shape, "type of w:", type(w))
K = np.array(sparse.csr_matrix.todense(X@w)) * np.array(sparse.csr_matrix.todense(Y))
K = sparse.csr_matrix(-np.array(sparse.csr_matrix.todense(Y)) * f(K))
result0 = X.T@K + Lamda * w
result0 = np.sqrt((result0.T @ result0)[0, 0])
print("r0:", result0)

# # # # # # # # # # # # # # # # # # # #
while True:
# for t in range(iter):
    # update w
    Yhat = X @ w
    delta = Yhat - Y
    K = np.array(sparse.csr_matrix.todense(X@w)) * np.array(sparse.csr_matrix.todense(Y))
    K = sparse.csr_matrix(-np.array(sparse.csr_matrix.todense(Y)) * f(K))
    result = X.T@K + Lamda * w
    # result = X.T @ delta * 2 / N + Lamda * w
    r = np.sqrt(result.T.dot(result)[0, 0])
    if r < epsilon * result0:
        print("Converge!")
        break
    # gradient of loss function
    w = w - learning_rate * result
    # update coefficient vector w
    # cost = (delta.T @ delta)[0, 0] / N + Lamda / 2 * (w.T @ w)[0, 0]
    print(r)
    # find and store the cost
    costs.append(r)
# # # # # # # # # # # # # # # # # # # #
# print out optimized w vector
print("final value of w:", w)
print("final value of cost function:", costs[-1])


# # # # # # # # # # # # # # # # # # # #
# draw plots
fig = plt.figure(figsize=(8, 6))
# plot the costs
plt.plot(costs, label='reduction in r')
plt.xlabel('number of iteration')
plt.ylabel('value of r')
plt.legend()
# plot prediction vs target
# plt.subplot(2, 1, 2)
# a = np.array(sparse.csr_matrix.todense(Y).T)
# b = np.array(sparse.csr_matrix.todense(Yhat).T)
# plt.plot(a, b, 'oc', alpha=0.5, label='standardized target vs. prediction')
# plt.xlabel('target of y')
# plt.ylabel('prediction of y')
target_file = r'D:\wkspacePY\STA 141C\HW1\Exercise2.png'
plt.savefig(target_file)

# # # # # # # # # # # # # # # # # # # #
# predict for test set
delta_test = np.array(sparse.csr_matrix.todense(test_X @ w)) * np.array(sparse.csr_matrix.todense(test_Y))
delta_train = np.array(sparse.csr_matrix.todense(X @ w)) * np.array(sparse.csr_matrix.todense(Y))
print("accuracy of prediction: %f" % float(len(np.where(delta_test > 0)[0])/n))
print("accuracy of training: %f" % float(len(np.where(delta_train > 0)[0])/N))



