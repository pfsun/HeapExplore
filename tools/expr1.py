#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 23:10:33 2017

@author: cspl
"""

#%% necessary libs
import os
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import json as js
from sklearn.metrics import accuracy_score as accu

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

#%% read data
f = open('data', 'r')
X = js.load(f)
f.close()
data_size = len(X)
#%%
if 0:
    seq = np.array(X[np.random.randint(len(X))][1])
    inds = np.where(seq == 0)
#    plt.stem(X[np.random.randint(len(X))][1][1:100])
    

#%% prepare data
train_test_split = 0.5

idx = np.random.permutation(data_size)

train_idx = idx[0:int(train_test_split*data_size)]
train_data_size = len(train_idx)
test_idx = idx[int(train_test_split*data_size)+1:-1]
test_data_size = len(test_idx)





#%% model construction
model = Sequential()
model.add(LSTM(100, input_shape=(None, 1), return_sequences=True))
#model.add(LSTM(100))
#model.add(TimeDistributed(Dense(100,  activation='softmax')))
#model.add(TimeDistributed(Dense(100,  activation='softmax')))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


#%% training
epoch = 10
#batch_size = 1
recurr = 200
for ep in range(epoch):
    id = 1
    for bt in train_idx:
        print('epoch #', ep, 'processing #', id)
        seq = X[bt][0]
        label = X[bt][1]
#        print('sequence length =', len(seq))        
        if len(seq)>recurr:
            X_train_size = int(len(seq)/recurr)
            
            
            X_train = seq[:X_train_size*recurr]# (batch_size, timesteps, input_dim)
            X_train = np.reshape(X_train, (recurr, X_train_size)).T
            X_train = np.expand_dims(X_train, axis=2)
            y_train = label[:X_train_size*recurr]
            y_train = keras.utils.to_categorical(y_train, 2)
            y_train = np.reshape(y_train, (X_train_size, recurr,  2), order='C')
            
            
    #        X_train = np.expand_dims(np.expand_dims(seq, axis=0), axis=2)
    #        y_train = keras.utils.to_categorical(label, 2)
    #        y_train = np.expand_dims(y_train, axis=0)
            
            model.train_on_batch(X_train, y_train) 
            
            
            
            if 0:
                X_validation = np.reshape(seq, (1, len(seq), 1))
                pdt_validation = model.predict_classes(X_validation)
                ac_validation = accu(pdt_validation.T, label)
                print('epoch #', id, 'processing #', id, 'validation accuracy =', ac_validation)
            
            
            
            id = id + 1
#%%
if 0:
    ind = np.random.randint(len(test_idx))
    seq_test = X[test_idx[ind]][0]
    label_test = X[test_idx[ind]][1]
    X_test = np.reshape(seq_test, (1, len(seq_test), 1))
    pdt = model.predict_classes(X_test)
    ac = accu(pdt.T, label_test)
    print(ac)
#%%
ac = 0
id = 1
for ind in test_idx:
    seq_test = X[ind][0]
    label_test = X[ind][1]
    X_test = np.reshape(seq_test, (1, len(seq_test), 1))
    pdt = model.predict_classes(X_test)
    accuracy = accu(pdt.T, label_test)
    print('accuracy for #', id, 'sample =', accuracy)
    ac = ac + accuracy
    id = id +1
ac_avg = ac/len(test_idx)
print('average test accuracy =', ac_avg)




