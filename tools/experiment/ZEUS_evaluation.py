#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 22:11:21 2017

@author: kamadanen
"""


import os
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import scipy.io as sio
import time

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
#from keras.layers.wrappers import TimeDistributed




#abspath = os.path.abspath(__file__)
#dname = os.path.dirname(abspath)
#os.chdir(dname)


## configurate data
prg = ['mat', 'qsort', 'new', 'gd', 'conv', 'dct', 'dijkstra', 'aes', 'pid', 'partfilt']
fea = ['spectrograms', 'time']
win = ['w50', 'w100', 'w150', 'w200', 'w250',]


program = prg[1]
feature = fea[0]
window = win[3]

if feature == 'spectrograms':
    feature_file = 'SSS'
else:
    feature_file = 'TTT'

program_dir = os.path.join('.', 'Data', program, feature)

paths = []
branches = os.listdir(program_dir)

if branches[0][0] == '.':
    branches.pop(0)
branches.pop(-1)
num_branch = len(branches)

for br in branches:
    paths.append(os.path.join(program_dir, br))


files = [] # the locations of all samples, each sublist one branch
for path in paths:
    files.append(os.listdir(path))
    
    
for i in range(num_branch):
    num_file = len(files[i])
    for j in range(num_file):
        files[i][j] = os.path.join('br'+str(i+1), files[i][j])
    

for i in range(num_branch):
    if files[i][0][4] == '.':
        files[i].pop(0)

files1 = sum(files, []) # single flat list of all samples for indexing
    

#y = np.array([])
file_indices = [] # each sublist indices for one branch
cumulate_num = 0 # cumulated number of files
for i in range(num_branch):
    num_file = len(files[i])
    file_indices.append(list(range(cumulate_num, cumulate_num+num_file)))
    cumulate_num += num_file
    



# tain&test split, shuffle within each class at each run
split = 0.5 # percentage of training samples, must be compatible with total number
file_indices_train = []
file_indices_test = []
for i in range(num_branch):
    num_file = len(file_indices[i])
    rd.shuffle(file_indices[i])
    file_indices_train.append(file_indices[i][:int(split*num_file)])
    file_indices_test.append(file_indices[i][int(split*num_file):])



# define network structure
timesteps = []
for br in range(num_branch):
    example = sio.loadmat(os.path.join(program_dir, files[br][0]))[feature_file]
    timesteps.append(example.shape[1])
dim = example.shape[0]

model = Sequential()
model.add(LSTM(100, input_shape=(None, dim), return_sequences=True))
model.add(LSTM(100))
#model.add(TimeDistributed(Dense(100,  activation='softmax')))
#model.add(TimeDistributed(Dense(100,  activation='softmax')))
model.add(Dense(num_branch,  activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])





# train network with minibatches
epoch = 20
mini_batch = 1
for ep in range(epoch):
    print('epoch =', ep)
    for branch in file_indices_train:
        rd.shuffle(branch)
    batch_size = int(len(file_indices_train[0])/mini_batch)
    for bt in range(mini_batch): # note elements in the following two arrays are float
        for br in range(num_branch):            
            print('batch =', bt, 'branch =', br) # for each batch and samples from each branch
#            print(np.array(range(batch_size*bt, batch_size*(bt+1))))
            y_train = br*np.ones(batch_size) # float type label
            y_train_cat = keras.utils.to_categorical(y_train, num_branch)
            X_train_indices = file_indices_train[br][batch_size*bt:batch_size*(bt+1)]
            # read training data for each batch
            X_train = np.array(np.zeros([batch_size, timesteps[br], dim]))
            for i in range(len(X_train_indices)):
                idx = int(X_train_indices[i])
                seq = sio.loadmat(os.path.join(program_dir, files1[idx]))[feature_file]
                X_train[i, :, :] = seq.T
                
                
            model.train_on_batch(X_train, y_train_cat)    
            
            
            score = model.predict_proba(X_train)
            score = np.max(score, axis=1)
            print('average belief is', np.mean(score))
        
        
#            pdt = model.predict_classes(X_train)
#            batch_acc = accuracy_score(pdt, y_train)
#            print('accuracy on batch is ', batch_acc)
#%% test network 

# predict class, compute acc, report belief score on each branch of the legitimate program, 
program_dir = os.path.join('.', 'Data', program, feature)
acc = 0
score = np.array([]);
for br in range(num_branch):
    num_test_br = len(file_indices_test[br])
    X_test = np.zeros([num_test_br, timesteps[br], dim])
    y_test = br*np.ones(num_test_br)
    for i in range(num_test_br):
        idx = int(file_indices_test[br][i])
        seq = sio.loadmat(os.path.join(program_dir, files1[idx]))[feature_file]
        X_test[i, :, :] = seq.T
    pdt = model.predict_classes(X_test)
    acc += accuracy_score(pdt, y_test)
    
    prob = model.predict_proba(X_test)
    score = np.append(score, np.max(prob, axis=1))
avg_acc = acc/num_branch
print('average accuracy on validation is', avg_acc)
#%% test mal1

mal_dir = os.path.join(program_dir, 'mal', 'mal1')

file_mal = os.listdir(mal_dir)

score_mal1 = []
for mal in file_mal:
    seq = sio.loadmat(os.path.join(mal_dir, mal))[feature_file].T
    seq = np.expand_dims(seq, axis=0)
    prob = model.predict_proba(seq)
    score_mal1 = np.append(score_mal1, np.max(prob))
#%% test mal2
mal_dir = os.path.join(program_dir, 'mal', 'mal2')

file_mal = os.listdir(mal_dir)

score_mal2 = []
for mal in file_mal:
    seq = sio.loadmat(os.path.join(mal_dir, mal))[feature_file].T
    seq = np.expand_dims(seq, axis=0)
    prob = model.predict_proba(seq)
    score_mal2 = np.append(score_mal2, np.max(prob))
#%%
def intrusionDetection(malware):
    import sklearn.metrics as sm
    from sklearn.metrics import accuracy_score
    
    mal_dir = os.path.join(program_dir, 'mal', malware)
    
    file_mal = os.listdir(mal_dir)
    
    score_mal = []
    for mal in file_mal:
        seq = sio.loadmat(os.path.join(mal_dir, mal))[feature_file].T
        seq = np.expand_dims(seq, axis=0)
        prob = model.predict_proba(seq)
        score_mal = np.append(score_mal, np.max(prob))
    # plot score distribution
    plt.figure()
    hist = np.histogram(score, bins=[x / 1000 for x in range(1000)])
    hist1 = np.histogram(score_mal, bins=[x / 1000 for x in range(1000)])
    plt.stem(hist[1][:-1], hist[0], 'b')
    plt.stem(hist1[1][:-1], hist1[0], 'r')
    # plot ROC curve and AUC
    score_target = np.append(score, score_mal)
    score_true = np.append(np.zeros(100), np.ones(100))
    plt.figure()
    fpr, tpr, th = sm.roc_curve(score_true, score_target, pos_label=0)
    auc = sm.roc_auc_score(score_true, 1-score_target)
    plt.plot(fpr, tpr)
    plt.title(auc)
    return
#%%
intrusionDetection('mal2')
#%% visualize results
# score distribution

plt.figure()
hist = np.histogram(score, bins=[x / 1000 for x in range(1000)])
hist1 = np.histogram(score_mal1, bins=[x / 1000 for x in range(1000)])
plt.stem(hist[1][:-1], hist[0], 'b')
plt.stem(hist1[1][:-1], hist1[0], 'r')

plt.figure()
hist = np.histogram(score, bins=[x / 1000 for x in range(1000)])
hist1 = np.histogram(score_mal2, bins=[x / 1000 for x in range(1000)])
plt.stem(hist[1][:-1], hist[0], 'b')
plt.stem(hist1[1][:-1], hist1[0], 'r') 

# roc curve
score1 = np.append(score, score_mal1)
score2 = np.append(score, score_mal2)

score_true = np.append(np.zeros(100), np.ones(100))

plt.figure()
fpr, tpr, th = sm.roc_curve(score_true, score1, pos_label=0)
auc = sm.roc_auc_score(score_true, 1-score1)
plt.plot(fpr, tpr)
plt.title(auc)

plt.figure()
fpr, tpr, th = sm.roc_curve(score_true, score2, pos_label=0)
auc1 = sm.roc_auc_score(score_true, 1-score2)
plt.plot(fpr, tpr)
plt.title(auc1)






