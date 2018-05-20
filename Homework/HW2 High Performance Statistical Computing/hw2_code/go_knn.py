"""
A few adjustments has been made for this module to run under python 3.6.
Redirected cPickle package and print function
"""
import pickle as cPickle
# cPickle is overloaded by pickle in python3
import multiprocessing as mp
import numpy as np
import time


fin = open("data_files.pl", "rb")
data = cPickle.load(fin,  encoding='iso-8859-1')
# default encoding = 'utf-8', encounter UnicodeDecodeError in python3
Xtrain = data[0]
ytrain = data[1]
Xtest = data[2]
ytest = data[3]



def go_nn(Xtrain, ytrain, Xtest, ytest):
    correct =0
    for i in range(Xtest.shape[0]): ## For all testing instances
        nowXtest = Xtest[i,:]
        ### Find the index of nearest neighbor in training data
        dis_smallest = np.linalg.norm(Xtrain[0,:]-nowXtest) 
        idx = 0
        for j in range(1, Xtrain.shape[0]):
            dis = np.linalg.norm(nowXtest-Xtrain[j,:])
            if dis < dis_smallest:
                dis_smallest = dis
                idx = j
        ### Now idx is the index for the nearest neighbor
        
        ## check whether the predicted label matches the true label
        if ytest[i] == ytrain[idx]:  
            correct += 1
    acc = correct/float(Xtest.shape[0])
    return acc

start_time = time.time()
acc = go_nn(Xtrain, ytrain, Xtest, ytest)
print("Accuracy %lf Time %lf secs.\n"%(acc, time.time()-start_time))

