# -*- coding : utf-8 -*-
"""
@author: Mingyi Xue
May/12/2018
Parallel Gradient Descent
This module is aimed at parallelize gradient descent solver for logistic regression
using multicore programming.
Most codes are copied directly from HW1/Exercise2.py, but capsuled in separate functions,
in order to be invoked in main function, because multiprocessing cannot be used in
an interactive environment.
"""
import numpy as np
import multiprocessing as mp
import datetime
from sklearn import datasets
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time


def load_data(filename, percent):
    """
    load dataset from filename, split dataset into training set and test set,
    return splited data
    :param filename: absolute file path
    :param percent: percentage of test set
    :return: X, test_X, Y, test_Y
    """
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
    print("split dataset into %d %d" % (int(percent*100), int((1-percent)*100)))
    X, test_X, Y, test_Y = train_test_split(X, Y, test_size=percent)
    return X, test_X, Y, test_Y


def initialize_dims(X, test_X):
    """
    return the shape of training set and test set
    :param X: X of training set
    :param test_X: X of test set
    :return: N, D, n, d
    """
    # initialize parameters
    N, D = X.shape
    # number of training data
    n, d = test_X.shape
    return N, D, n, d


def get_hyperpars():
    """
    Hyperparameters
    :return: learning_rate, Lamda, iter, epsilon
    """
    learning_rate = 1.0 * 10 ** (-2)
    # step size
    Lamda = 1
    iter = 5000
    # number of iteration
    epsilon = 0.001
    # converge condition
    return learning_rate, Lamda, iter, epsilon


def initialize_parameters(D):
    """
    Initialize vector w using np.random.randn
    :param D: number of features
    :return: w
    """
    # initialize vector of omega
    w = np.random.randn(D)
    w = sparse.csr_matrix(np.matrix(w)).T
    print("shape of w:", w.shape, "type of w:", type(w))
    return w


def draw_plots(output_file, costs):
    """
    Draw reduction in cost function.
    :param output_file: save path
    :param costs: a list of cost value of each iteration
    :return: None
    """
    # draw plots
    plt.figure(figsize=(8, 6))
    # plot the costs
    plt.plot(costs, label='reduction in r')
    plt.xlabel('number of iteration')
    plt.ylabel('value of r')
    plt.legend()
    plt.savefig(output_file)


def get_accuracy(X, Y, test_X, test_Y, w):
    """
    Calculate accuracy on training set and test set.
    :param X: X of training set
    :param Y: Y of training set
    :param test_X: X of test set
    :param test_Y: Y of test set
    :param w: parameter, vector w
    :return: None
    """
    # predict for test set
    delta_test = np.array(sparse.csr_matrix.todense(test_X @ w)) * np.array(sparse.csr_matrix.todense(test_Y))
    delta_train = np.array(sparse.csr_matrix.todense(X @ w)) * np.array(sparse.csr_matrix.todense(Y))
    print("accuracy of prediction: %f" % float(len(np.where(delta_test > 0)[0])/n))
    print("accuracy of training: %f" % float(len(np.where(delta_train > 0)[0])/N))


## define a iterate function for matrix,
## apply this function to each element in the matrix
def f(x):
    """
    Broadcast function
    :param x: np.ndarray or np.matrix
    :return: 1/(1+np.exp(x))
    """
    return 1/(1+np.exp(x))


def compute_r(t, w, Lamda):
    """
    Compute r for each iteration,
    the algorithm reaches convergence and stops when r < epsilon * r0
    :param t: tuple(X, Y)
    :param w: parameter, vector w
    :param Lamda: hyperparameter, Lamda
    :return: (vector)result
    """
    X, Y = t
    K = np.array(sparse.csr_matrix.todense(X @ w)) * np.array(sparse.csr_matrix.todense(Y))
    K = sparse.csr_matrix(-np.array(sparse.csr_matrix.todense(Y)) * f(K))
    result = X.T @ K
    # print("type of result:", type(result), "shape of result:", result.shape)
    return result


def slice(X, Y, n):
    """
    Divide X, Y evenly into n parts
    :param X:matrix or array
    :param Y: matrix or array
    :param n:number of parts
    :return: a list of slices
    """
    lst = []
    size = np.ceil(X.shape[0] / n)
    for i in range(n):
        if (i + 1) * size > X.shape[0]:
            lst.append((X[int(i * size):, :], Y[int(i * size):]))
        else:
            lst.append((X[int(i * size):int((i + 1) * size), :],
                        Y[int(i * size):int((i + 1) * size)]))
    return lst


if __name__ == "__main__":
    ## initialization
    filename = r'C:\Users\薛明怡\Desktop\2018 spring quarter\STA 141C\Homework\data\news20.binary.bz2'
    output_file = r'D:\wkspacePY\STA 141C\HW2\Exercise2.png'
    percent = 0.2
    X, test_X, Y, test_Y = load_data(filename, percent)
    N, D, n, d = initialize_dims(X, test_X)
    learning_rate, Lamda, iter, epsilon = get_hyperpars()
    w = initialize_parameters(D)
    costs = []
    process_num = 4
    ## iteration
    ## note that while iteration cannot be parallelized
    # because r depends on w computed from last iteration
    lst = slice(X, Y, process_num)
    start_time = time.time()
    result = 0
    with mp.Pool(process_num) as pool:
        result = pool.starmap(compute_r, [(l, w, Lamda) for l in lst])
    result = np.array(result)
    print("length of result:", len(result))
    result = np.sum(result) + Lamda * w
    r0 = np.sqrt(result.T.dot(result)[0, 0])
    print("r0:", r0)
    while True:
        with mp.Pool(process_num) as pool:
            result = pool.starmap(compute_r, [(l, w, Lamda) for l in lst])
        result = np.array(result)
        result = np.sum(result) + Lamda * w
        r = np.sqrt(result.T.dot(result)[0, 0])
        if r < epsilon * r0:
            print("Converge!")
            break
        # update coefficient vector w
        w = w - learning_rate * result
        print(r)
        # find and store the cost
        costs.append(r)
    # print out optimized w vector
    end_time = time.time()
    pool.close()
    print("final value of cost function:", costs[-1])
    print("Time cost %lf secs." % (end_time - start_time))
    get_accuracy(X, Y, test_X, test_Y, w)
    # draw plots
    draw_plots(output_file, costs)
