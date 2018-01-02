#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 17:01:38 2017

@author: cspl

train on 5 images of length 60k, manually reshaped batches
"""

#%% necessary libs
#import numpy as np
#import matplotlib.pyplot as plt
#import json as js
#import data_visualization as dv
#from sklearn.metrics import accuracy_score as accu
from keras import optimizers as opt
#import keras as krs
import heapexplore_utils as utils

from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, GRU

root = '../../../'
data_path = root + 'Data/data'
data_idx_path = root + 'Data/idx'
result_path = root + 'results/model.h5'
#%% read data
X = utils.loadData(data_path)  
#%%
recurr = 1000
cat = 3
XX, yy = utils.reshapeData(X[3:8], recurr, 4000)
X_train, y_train = utils.label_to_cat(XX, yy, cat)
#%% model construction
adam = opt.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model = Sequential()
model.add(LSTM(100, batch_input_shape=(5, None, 1), return_sequences=True, stateful=True))
model.add(LSTM(100, return_sequences=True, stateful=True))
model.add(LSTM(100, return_sequences=True, stateful=True))
#model.add(LSTM(100))
#model.add(TimeDistributed(Dense(1,  activation='sigmoid')))
#model.add(TimeDistributed(Dense(100,  activation='softmax')))
#model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
#%% training
#model.reset_states()
model.fit(X_train, y_train, epochs=50, batch_size=5, shuffle=False)
model.save(result_path)
#%% validation on training sequence
def test_on_heap(X, sample_idx):
    X_validation = np.reshape(X[sample_idx][0], (1, len(X[sample_idx][0]), 1))
    pdt = model.predict_classes(X_validation, batch_size=1)
    accuracy = accu(pdt[:, :, 0].T, X[sample_idx][1])
    print('training accuracy is', accuracy)
    
#%%
test_on_heap(X, 1   )