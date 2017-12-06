#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 00:27:35 2017

@author: cspl

multiple mem imgs, with stateful

"""

#%% necessary libs
import numpy as np
import matplotlib.pyplot as plt
import json as js
import data_visualization as dv

from sklearn.metrics import accuracy_score as accu
from keras import optimizers as opt

from keras.models import Sequential
from keras.layers import Dense, LSTM#, TimeDistributed

adam = opt.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
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
model.add(LSTM(10, batch_input_shape=(1, None, 1), return_sequences=True, stateful=False))
model.add(LSTM(10, return_sequences=True, stateful=False))
model.add(LSTM(10, return_sequences=True, stateful=False))
#model.add(LSTM(100))
#model.add(TimeDistributed(Dense(1,  activation='sigmoid')))
#model.add(TimeDistributed(Dense(100,  activation='softmax')))
#model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
#%% training
epoch = 10
#batch_size = 1
recurr = 20
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
            y_train = np.reshape(y_train, (X_train_size, recurr))
            y_train = np.expand_dims(y_train, axis=2)
            
            ls = 0;
            ac = 0
            for sample_idx in range(X_train.shape[0]):
                X_train_sample = X_train[sample_idx, :, :]
                y_train_sample = y_train[sample_idx, :, :]
                output = model.train_on_batch(np.expand_dims(X_train_sample, axis=0), np.expand_dims(y_train_sample, axis=0))
                ls += output[0]
                ac += output[1]
            ls = ls / X_train.shape[0]
            ac = ac / X_train.shape[0]
            print('epoch', ep, '/', epoch, ', sample', id, '/', len(train_idx), ',', model.metrics_names[0], ls, ',',  model.metrics_names[1], ac)
                
            model.reset_states()
            id = id + 1
model.save('../results/model.h5')
#%% validation on training sequence
X_validation = np.reshape(X[sample_idx][0], (1, len(X[sample_idx][0]), 1))
pdt = model.predict_classes(X_validation)
accuracy = accu(pdt[:, :, 0].T, X[sample_idx][1])
print('training accuracy is', accuracy)
#%% validation result plot
plt.figure(figsize=(12, 3))
plt.subplot(121)
plt.plot(pdt[:, :, 0].T)
plt.title('prediction')
plt.subplot(122)
plt.plot(X[sample_idx][1])
plt.title('truth')

plt.figure(figsize=(12, 3))
plt.subplot(121)
plt.plot(pdt[:, :, 0].T)
plt.title('prediction')
plt.xlim([8000, len(pdt[:, :, 0].T)])
plt.subplot(122)
plt.plot(X[sample_idx][1])
plt.title('truth')
plt.xlim([8000, len(X[sample_idx][1])])
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
    accuracy = accu(pdt[:, :, 0].T, label_validation)
    print('accuracy for #', id, '/', len(validation_idx), 'sample =', accuracy, 'sample length =', len(seq_validation))
    ac = ac + accuracy
    id = id +1
ac_avg = ac/len(validation_idx)
print('****************************************************************')
print('average validation accuracy =', ac_avg)
#%% visualize data
vasualization_idx = np.arange(9)
offset = 9

dv.plotData(vasualization_idx+offset, X)
#%% visualize prediction results
vasualization_idx = np.arange(9)
offset = 9

dv.plotResult(vasualization_idx+offset, result, truth)