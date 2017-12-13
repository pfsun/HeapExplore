#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 23:10:33 2017

@author: kamadanen 

batch size 

Note: please put data at root before run the experiment
"""

#%% necessary libs
import numpy as np
#import random as rd
import matplotlib.pyplot as plt
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
if 0:
    train_test_split = 0.5
    
    idx = np.random.permutation(data_size)
    
    train_idx = idx[0:int(train_test_split*data_size)]
    train_data_size = len(train_idx)
    test_idx = idx[int(train_test_split*data_size)+1:-1]
    test_data_size = len(test_idx)
    data_idx = np.array([train_idx, test_idx])
    np.save('../Data/idx', data_idx)
#%% model construction
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), return_sequences=True,
    activation='relu'))
#model.add(LSTM(100))
#model.add(TimeDistributed(Dense(100,  activation='softmax')))
#model.add(TimeDistributed(Dense(100,  activation='softmax')))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
#%% training
epoch = 50
recurr = 200
for ep in range(epoch):
    id = 1
    for bt in train_idx:
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
            
            
    #        X_train = np.expand_dims(np.expand_dims(seq, axis=0), axis=2)
    #        y_train = keras.utils.to_categorical(label, 2)
    #        y_train = np.expand_dims(y_train, axis=0)
            
            loss = model.train_on_batch(X_train, y_train) 
            print('epoch #', ep, 'processing #', id, model.metrics_names[0], loss[0], model.metrics_names[1], loss[1])
            
            if 0:
                X_validation = np.reshape(seq, (1, len(seq), 1))
                pdt_validation = model.predict_classes(X_validation)
                ac_validation = accu(pdt_validation.T, label)
                print('epoch #', id, 'processing #', id, 'validation accuracy =', ac_validation)
            
            
            
            id = id + 1
model.save('../results/model.h5')

#%% validation on training set
ac = 0
id = 0
percent = 0.5
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
    print('accuracy for #', id, '/', len(validation_idx), 'sample =', accuracy, 'sample length =', len(seq_validation))
    ac = ac + accuracy
    id = id +1
ac_avg = ac/len(validation_idx)
print('****************************************************************')
print('average validation accuracy =', ac_avg)
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
for i in range(10):
    plt.figure()
    plt.plot(X_train[i, :, :])
#%%
offset = 70
plt.figure(figsize=(12, 18))
for i in range(1+offset, 10+offset):
    plt.subplot(520+i-offset)
    plt.plot(result[i][0, :])
    plt.title(i)
#%%
offset = 5
plt.figure(figsize=(12, 18))
for i in range(1+offset, 10+offset):
    plt.subplot(520+i-offset)
    plt.plot(truth[i])
    plt.title(i)



