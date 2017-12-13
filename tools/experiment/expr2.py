#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:46:24 2017

@author: cspl

to use stateful lstm with batch size 1
"""

#%% necessary libs
import numpy as np
#import random as rd
#import matplotlib.pyplot as plt
#import scipy.io as sio
import json as js
from sklearn.metrics import accuracy_score as accu

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

#%% read data
f = open('../Data/data', 'r')
X = js.load(f)
f.close()
data_size = len(X)    
#%% prepare data
train_test_split = 0.1

idx = np.random.permutation(data_size)

train_idx = idx[0:int(train_test_split*data_size)]
train_data_size = len(train_idx)
test_idx = idx[int(train_test_split*data_size)+1:-1]
test_data_size = len(test_idx)
data_idx = np.array([train_idx, test_idx])
np.save('../Data/idx', data_idx)
#%% model construction
model = Sequential()
model.add(LSTM(2, input_shape=(None, 1), return_sequences=True,
    activation='softmax'))
#model.add(LSTM(100))
#model.add(TimeDistributed(Dense(100,  activation='softmax')))
#model.add(TimeDistributed(Dense(100,  activation='softmax')))
#model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
#%% training
epoch = 1
#batch_size = 1
recurr = 20
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
            X_train = np.reshape(X_train, (X_train_size, recurr))
            X_train = np.expand_dims(X_train, axis=2)
            y_train = label[:X_train_size*recurr]
            y_train = keras.utils.to_categorical(y_train, 2)
            y_train = np.reshape(y_train, (X_train_size, recurr,  2), order='C')
            
            for sample_idx in range(X_train.shape[0]):
                X_train_sample = X_train[sample_idx, :, :]
                y_train_sample = y_train[sample_idx, :, :]
                model.train_on_batch(X_train, y_train)
            
    #        X_train = np.expand_dims(np.expand_dims(seq, axis=0), axis=2)
    #        y_train = keras.utils.to_categorical(label, 2)
    #        y_train = np.expand_dims(y_train, axis=0)
            
#            model.train_on_batch(X_train, y_train) 
            
            
            
            
            id = id + 1
model.save('../results/model.h5')
#%% validation on training set
ac = 0
id = 0
percent = 0.1
validation_idx = train_idx[:int(len(train_idx)*percent)]
truth = []
result = []
for ind in validation_idx:
    seq_validation = X[ind][0]
    label_validation = X[ind][1]
    truth.append(label_validation)
    X_validation = np.reshape(seq_validation, (1, len(seq_validation), 1))
    pdt = model.predict_classes(X_validation)
    result.append(pdt)
    accuracy = accu(pdt.T, label_validation)
    print('accuracy for #', id, 'sample =', accuracy, 'sample length =', len(seq_validation))
    ac = ac + accuracy
    id = id +1
ac_avg = ac/len(validation_idx)
print('****************************************************************')
print('average validation accuracy =', ac_avg)