# -*- coding: utf-8 -*-
"""
@author: Mingyi Xue
May/28/2018
K-means clustering
sparse dataset which can be downloaded from
http://www.stat.ucdavis.edu/~chohsieh/teaching/STA141C_Spring2018/hw3_data.zip.
data_sparse_E2006.pl
difference from kmeans_dense:
1. add the computation of smallest cost directly to j
2. 'centers' is designed to be a list in order to save variable assignment time
3. use generator instead of explicit for-loop to save iteration time
"""


import pickle as cPickle
# cPickle is overloaded by pickle in python3
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from scipy import sparse
from scipy.sparse import linalg


def read_data(filename):
    """
    load data from filename
    :param filename: dataset source
    :return: data
    """
    print("Begin to load data from:", filename)
    start = time.time()
    fin = open(filename, "rb")
    data = cPickle.load(fin, encoding='latin1')
    finish = time.time()
    print("Dataset load finished, time cost:%lf secs" % (finish - start))
    print("shape of training data:", data.shape, "type:", type(data))
    return data


def initial_center(X, num):
    """
    choose the first <num> samples from dataset as initial centers
    :param X: data
    :param num: number of centers
    :return: list of centers
    """
    centers = [X[i, :] for i in range(num)]
    print("info of centers:", type(centers), len(centers))
    return centers




def cluster(X, centers, iter=40, draw=False, plot="1.png"):
    """
    k means cluster
    :param X: data
    :param centers: k initialized centers
    :param iter: number of iteration
    :param draw: draw plots
    :return: centers, dict{center_index: list of index}
    """
    dit = {}
    J = []
    time_lst = []
    time_lst.append(time.time())
    for it in range(iter):
        j = 0
        for i in range(len(centers)):
            dit[i] = []
        for i in range(X.shape[0]):
            tmp = np.array([((X[i, :] - centers[j])@(X[i, :] - centers[j]).T)[0, 0] for j in range(len(centers))])
            current = np.argmin(tmp)
            dit[current].append(i)
            j += tmp[current]
        print("J in %d th iteration: %lf" % (it+1, j))
        J.append(j)
        centers = [sparse.csr_matrix(np.sum(X[v, :], axis=0) / len(v)) for k, v in dit.items()]
        print("info of centers:", type(centers), len(centers))
        time_lst.append(time.time())
    for i in range(iter):
        if i % 10 == 9:
            print("J of the %d th iteration:" % (i+1), J[i])
            print("total time cost: %lf secs" % (time.time() - time_lst[0]))
    if draw:
        plot_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), plot)
        draw_plot(plot_file, J)
    return centers, dit


def draw_plot(filename, J):
    """
    visualize the reduction of k-means cluster
    :param filename: save plot filename
    :param J: k-means objective
    :return: None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(J, label='reduction in j')
    plt.xlabel('number of iteration')
    plt.ylabel('value of j')
    plt.legend()
    plt.savefig(filename)


if __name__ == "__main__":
    filename = r'C:\Users\薛明怡\Desktop\2018 spring quarter\STA 141C\Homework\data\data_sparse_E2006.pl'
    X = read_data(filename)
    centers = initial_center(X, 10)
    centers, dit = cluster(X, centers, iter=40, draw=True, plot="3.png")







