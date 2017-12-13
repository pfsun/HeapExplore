#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:07:42 2017

@author: cspl

single mem img, with stateful

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:46:24 2017

@author: cspl

to train on one whole image
"""

#%% necessary libs
import numpy as np
import matplotlib.pyplot as plt
import json as js
import data_visualization as dv
from sklearn.metrics import accuracy_score as accu
from keras import optimizers as opt
import keras as krs

from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, GRU

adam = opt.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

data_path = '../../Data/data'
data_idx_path = '../../Data/idx'
result_path = '../../results/model.h5'
#%% read data
f = open(data_path, 'r')
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
np.save(data_idx_path, data_idx)
#%% model construction
model = Sequential()
model.add(LSTM(100, batch_input_shape=(1, None, 1), return_sequences=True, stateful=True))
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
sample_idx = 4 #train_idx[0]


recurr = 200
seq = X[sample_idx][0]
label = X[sample_idx][1]


print('training sample length =', len(seq))
    
X_train = np.expand_dims(np.expand_dims(seq, axis=2), axis=0)
#y_train = np.expand_dims(np.expand_dims(label, axis=2), axis=0)
y_train = krs.utils.to_categorical(label, 3)
    
    model.reset_states()
    model.fit(X_train[:, :, :], y_train[:, :, :], epochs=50, batch_size=1, shuffle=False)
model.save(result_path)
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
#%%
offset = 0
plt.figure(figsize=(12, 8))
for i in range(0, 9):
    plt.subplot(331+i)
    plt.plot(X_train[i+offset, :, :])
    plt.ylim([-10, 260])
    plt.title(i+offset)
#%%
offset = 0
plt.figure(figsize=(12, 8))
for i in range(0, 9):
    plt.subplot(331+i)
    plt.plot(y_train[i+offset, :, :])
    plt.ylim([-0.1, 1.1])
    plt.title(i+offset)
#%%
offset = 9
plt.figure(figsize=(12, 8))
for i in range(0, 9):
    plt.subplot(331+i)
    plt.plot(result[i+offset][0, :])
    plt.title(i+offset)

plt.figure(figsize=(12, 8))
for i in range(0, 9):
    plt.subplot(331+i)
    plt.plot(truth[i+offset])
    plt.title(i+offset)