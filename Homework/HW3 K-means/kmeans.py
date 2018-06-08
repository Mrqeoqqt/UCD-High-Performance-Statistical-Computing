# -*- coding: utf-8 -*-
"""
@author: Mingyi Xue
May/28/2018
K-means clustering
This module aims at implementing the "k-means" algorithm to cluster datasets.

dense dataset data_dense.pl, which can be downloaded from
http://www.stat.ucdavis.edu/~chohsieh/teaching/STA141C_Spring2018/hw3_data.zip.
"""

import pickle as cPickle
# cPickle is overloaded by pickle in python3
import numpy as np
import time
import matplotlib.pyplot as plt
import os


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


def initial_center_rand(X, num):
    """
    randomly choose <num> samples from dataset as initial centers
    :param X: data
    :param num: number of centers
    :return: list of centers
    """
    row_num = X.shape[0]
    shuffle = np.arange(row_num)
    np.random.shuffle(shuffle)
    centers = X[shuffle[:num], :]
    return centers

def initial_center(X, num):
    """
    choose the first <num> samples from dataset as initial centers
    :param X: data
    :param num: number of centers
    :return: list of centers
    """
    centers = X[:num, :]
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
    start = time.time()
    for it in range(iter):
        oj = 0
        for i in range(centers.shape[0]):
            dit[i] = []
        for i in range(X.shape[0]):
            current = 0
            cost = np.linalg.norm(np.array(X[i, :]-centers[0, :]))
            for j in range(1, centers.shape[0]):
                tmp = np.linalg.norm(np.array(X[i, :]-centers[j, :]))
                if tmp < cost:
                    cost = tmp
                    current = j
            dit[current].append(i)
            oj += cost**2
        print("J in %d iteration:" % (it+1), oj)
        J.append(oj)
        for k, v in dit.items():
            v = X[v, :]
            v = np.sum(v, axis=0)/v.shape[0]
            centers[k] = v
    print("clustering time cost:%lf secs" % (time.time()- start))
    for i in range(iter):
        if i % 10 == 9:
            print("J of the %d th iteration:" % (i+1), J[i])
    if draw:
        plot_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), plot)
        draw_plot(plot_file, J)
    return centers, dit


def compute_J(X, dit, centers):
    """
    J is a criteria of how well the model fits
    :param X: data
    :param centers: k centers
    :return: J
    """
    j = 0
    for k, v in dit.items():
        j += np.sum(np.linalg.norm(X[v, :] - centers[k, :])**2)
    return j


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
    filename = r'C:\Users\薛明怡\Desktop\2018 spring quarter\STA 141C\Homework\data\data_dense.pl'
    X = read_data(filename)
    centers = initial_center_rand(X, 10)
    centers, dit = cluster(X, centers, iter=40, draw=True, plot="1.png")


