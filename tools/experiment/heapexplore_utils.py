#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:24:57 2017

@author: cspl
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

#%% distribution of the heap sizes
def plotHist(X):
    ll = []
    for sample in X:
        ll.append(len(sample[0]))
    ll_hist = np.histogram(ll, bins=1000)
    plt.figure()
    plt.plot(ll_hist[1][:-1], ll_hist[0])

#%% visualize data, 9 dump at a time
def visualizeData(X, offset):
    vasualization_idx = np.arange(9) 
    dv.plotData(vasualization_idx+offset, X)   
    plt.show()

#%% to extract dumps with a specific size range
def extractDump(X, lower, upper):
    X1 = []
    for sample in X:
        if len(sample[0]) < upper:
            if len(sample[0]) > lower:
                X1.append(sample)
    return X1

#%% to prepare input data
def reshapeData(Data, recurr):
    seq = Data[0]
    label = Data[1]
    ll = len(seq)
    num_subseq = int(ll/recurr)
    seq1 = seq[]
