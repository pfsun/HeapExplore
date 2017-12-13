#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 00:21:17 2017

@author: cspl

explore data

"""

import data_visualization as dv
import numpy as np
import matplotlib.pyplot as plt
import json as js
import keras



#%% read data
def loadData(directory):
    f = open(directory, 'r')
    X = js.load(f)
    f.close()
    return X

#%%
def plotHist(X):
    ll = []
    for sample in X:
        ll.append(len(sample[0]))
    ll_hist = np.histogram(ll, bins=1000)
    plt.figure()
    plt.plot(ll_hist[1][:-1], ll_hist[0])

#%% visualize data
def visualizeData(X, offset):
    vasualization_idx = np.arange(9) 
    dv.plotData(vasualization_idx+offset, X)   
    plt.show()
#%%
def extractDump(X, lower, upper):
    X1 = []
    for sample in X:
        if len(sample[0]) < upper:
            if len(sample[0]) > lower:
                X1.append(sample)
    return X1
##%%
#max_len = 0
#for sample in X1:
#    if max_len < len(sample[0]):
#        max_len = len(sample[0])
##%%
##data = np.zeros([len(X1), max_len])
#X_train_list = []
#y_train_list = []
#for sample in X1:
#    X_train_list.append(sample[0])
#    y_train_list.append(sample[1])
#X_train = keras.preprocessing.sequence.pad_sequences(X_train_list, padding='post', value=0.)
#y_train = keras.preprocessing.sequence.pad_sequences(y_train_list, padding='post', value=0.)