#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:46:24 2017

@author: cspl

to train on several images, one at a time
"""

#%% necessary libs
import numpy as np
import matplotlib.pyplot as plt
import json as js
import heapexplore_utils as utils
from sklearn.metrics import accuracy_score as accu
from keras import optimizers as opt
import keras as krs

from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, GRU

root = '../../../'
data_path = root + 'Data/data'
data_idx_path = root + 'Data/idx'
result_path = root + 'results/model.h5'
#%% read data
X = utils.loadData(data_path)    
#%% model construction
adam = opt.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
stf = True
model = Sequential()
model.add(LSTM(100, batch_input_shape=(1, None, 1), return_sequences=True, stateful=stf))
model.add(LSTM(100, return_sequences=True, stateful=stf))
model.add(LSTM(100, return_sequences=True, stateful=stf))
#model.add(LSTM(100))
#model.add(TimeDistributed(Dense(1,  activation='sigmoid')))
#model.add(TimeDistributed(Dense(100,  activation='softmax')))
#model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
#%% prepare training data 
train_data = []
train_label = []
recurr = 200
cat = 3
for i in range(3, 4): # adjust training images here
    XX, yy = utils.reshapeData(X[i], recurr, None) # adjust image size here
    X_train, y_train = utils.label_to_cat(XX, yy, cat)
    train_data.append(X_train)
    train_label.append(y_train)
#%%
epoch = 50
for ep in range(epoch):
    for i in range(len(train_data)):
        for j in range(train_data[i].shape[0]):
            loss = model.train_on_batch(train_data[i][None, j, :, :], train_label[i][None, j, :, :])
        print('epoch #', ep, 'processing #', i, model.metrics_names[0], loss[0], model.metrics_names[1], loss[1])
    model.reset_states()
model.save(result_path)

#%% validation on training sequence
seq = X[2][0]
label = X[2][1]
X_validation = np.reshape(seq, (1, len(seq), 1))
pdt = model.predict_classes(X_validation)
plt.plot(pdt.T)
plt.plot(label)
plt.imshow()
accuracy = accu(pdt.T, label)
print('training accuracy is', accuracy)