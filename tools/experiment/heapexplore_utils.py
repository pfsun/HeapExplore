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
import keras as krs



#%% read data
def loadData(directory):
    f = open(directory, 'r')
    X = js.load(f)
    f.close()
    return X

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
def reshapeData(Data, recurr, ll):
    seq = []
    label = []
    if type(Data[0][0]) == list:
        for sample in Data:
            seq.append(sample[0])
            label.append(sample[1])
        
    else:
        seq = Data[0]
        label = Data[1]
    seq1 = np.array(seq)
    label1 = np.array(label)

    if len(seq1.shape) == 1:
        seq1 = np.expand_dims(seq1, axis=0)
        label1 = np.expand_dims(label1, axis=0)

    if ll==None:
        ll = seq1.shape[1]
        
    bt_size = seq1.shape[0]
    num_subseq = int(ll/recurr)
    ll1 = num_subseq*recurr
    
    seq2 = seq1[:, :ll1]
    label2 = label1[:, :ll1]
    print(seq2.shape)
    
    X_train = np.zeros([bt_size*num_subseq, recurr])
    y_train = np.zeros([bt_size*num_subseq, recurr])
    print(X_train.shape)
    for i in range(num_subseq):
        X_train[i*bt_size:(i+1)*bt_size, :] = seq2[:, i*recurr:(i+1)*recurr]
        y_train[i*bt_size:(i+1)*bt_size, :] = label2[:, i*recurr:(i+1)*recurr]
    
    return X_train, y_train
#%%
def label_to_cat(X, y, cat):
    X_train = np.expand_dims(X, axis=2)
    y_train = np.zeros((y.shape[0], y.shape[1], cat))
    for i in range(y.shape[0]):
        y_train[i, :, :] = krs.utils.to_categorical(y[i, :], cat)
    return X_train, y_train