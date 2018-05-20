# -*- coding: iso-8859-1 -*-
"""
@author: Mingyi Xue
May/12/2018
Multicore Programming
This module aims at modifying the "go_nn" function, which can be downloaded from
http://www.stat.ucdavis.edu/~chohsieh/teaching/STA141C_Spring2018/hw2_code.zip,
to parallelize the computation using multiple cores.
"""

import pickle as cPickle
# cPickle is overloaded by pickle in python3
import multiprocessing as mp
import numpy as np
import time


### load data
filename = r"D:\wkspacePY\STA 141C\HW2\hw2_code\data_files.pl"
fin = open(filename, "rb")
data = cPickle.load(fin, encoding='iso-8859-1')
# default encoding = 'utf-8', encounter UnicodeDecodeError in python3
Xtrain = data[0]
ytrain = data[1]
Xtest = data[2]
ytest = data[3]


### define go_knn function
def f(Xtrain, ytrain, Xtest, ytest):
    """
    Use knn algorithm to determine the label for sample Xtest
    and compare the label with ytest.
    If consistant, return 1, else return 0.
    :param Xtest: one test sample
    :param ytest: y value of the test sample
    :return: 1/0
    """
    ## Xtrain and ytrain are declared as global variables
    ## to reduce parameter passing time
    nowXtest = Xtest
    ## Find the index of nearest neighbor for a single Xtest
    dis_smallest = np.linalg.norm(Xtrain[0, :] - nowXtest)
    idx = 0
    for i in range(1, Xtrain.shape[0]):
        dis = np.linalg.norm(nowXtest - Xtrain[i, :])
        if dis < dis_smallest:
            dis_smallest = dis
            idx = i
    ## check whether the predicted label matches the true label
    ## if predicted label is right, return True, else return False
    if ytest == ytrain[idx]:
        return 1
    else:
        return 0


def h(Xtrain, ytrain, Xtest, ytest):
    """
    Use knn algorithm to determine the label for sample Xtest
    and compare the label with ytest.
    If consistant, queue.put(1)
    :param Xtest: test samples
    :param ytest: y values of input test samples
    :return: count
    """
    ## Xtrain and ytrain are declared as global variables
    ## to reduce parameter passing time
    count = 0
    for i in range(Xtest.shape[0]):  ## For all testing instances
        nowXtest = Xtest[i, :]
        ## Find the index of nearest neighbor in training data
        dis_smallest = np.linalg.norm(Xtrain[0, :] - nowXtest)
        idx = 0
        for j in range(1, Xtrain.shape[0]):
            dis = np.linalg.norm(nowXtest - Xtrain[j, :])
            if dis < dis_smallest:
                dis_smallest = dis
                idx = j
        ## check whether the predicted label matches the true label
        ## if predicted label is right, return True, else return False
        if ytest[i] == ytrain[idx]:
            count += 1
    return count


if __name__ == '__main__':
    mp.freeze_support()
    ## multiprocessing can only be invoked in main function

    ## use Pool
    ## line by line
    print("Parallel row by row using Pool...")
    start_time = time.time()
    pool = mp.Pool(processes=4)
    # record start time
    result = pool.starmap(f, [(Xtrain, ytrain, x, y) for (x, y) in zip(Xtest, ytest)])
    pool.close()
    # record finish time
    end_time = time.time()
    # calculate accuracy
    acc = sum(result)/len(result)
    # print result and info
    print("Accuracy %lf Time %lf secs.\n" % (acc, end_time - start_time))

    ## use Pool
    ## by chunk
    print("Parallel by chunk using Pool...")
    X = []
    Y = []
    process_num = 4
    size = np.ceil(len(Xtest) / process_num)
    for i in range(process_num):
        if (i + 1) * size > len(Xtest):
            X.append(Xtest[int(i * size):, :])
            Y.append(ytest[int(i * size):])
        else:
            X.append(Xtest[int(i * size):int((i + 1) * size), :])
            Y.append(ytest[int(i * size):int((i + 1) * size)])
    start_time = time.time()
    pool = mp.Pool(processes=4)
    result = pool.starmap(h, [(Xtrain, ytrain, x, y) for (x, y) in zip(X, Y)])
    pool.close()
    end_time = time.time()
    print("Accuracy %lf Time %lf secs.\n" %
          (np.sum(np.array(result))/Xtest.shape[0], end_time - start_time))