#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 00:21:17 2017

@author: cspl

explore data

"""

import data_visualization as dv
import numpy as np
import matplotlib.pyplot as plt
import json as js



#%% read data
f = open('../Data/data', 'r')
X = js.load(f)
f.close()
data_size = len(X) 
#%%
ll = []
for sample in X:
    ll.append(len(sample[0]))
ll_hist = np.histogram(ll, bins=1000)
plt.figure()
plt.plot(ll_hist[1][:-1], ll_hist[0])

#%% visualize data
vasualization_idx = np.arange(9)
offset = 100

dv.plotData(vasualization_idx+offset, X)


plt.show()