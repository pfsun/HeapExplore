#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 00:10:38 2017

@author: cspl
"""
#%%
from keras.models import load_model as ld
import numpy as np
import json as js
from sklearn.metrics import accuracy_score as accu
#%% read data
f = open('../Data/data', 'r')
X = js.load(f)
f.close()
data_size = len(X) 
#%%
data_idx = np.load('../Data/idx.npy')
test_idx = data_idx[1]
#%%
model = ld('../results/model.h5')
#%%
ac = 0
id = 1
percent = 0.1
test_idx = test_idx[:int(len(test_idx)*percent)]
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
