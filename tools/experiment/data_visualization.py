#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 14:51:42 2017

@author: cspl
"""
import matplotlib.pyplot as plt


def plotData(idx, X):
    plt.figure(figsize=(12, 12))
    i = 0
    for img in idx:
        plt.subplot(331+i)
        plt.plot(X[img][0])
        plt.ylim([-10, 260])
        plt.title(img)
        i += 1
    
    plt.figure(figsize=(12, 12))
    i = 0
    for img in idx:
        plt.subplot(331+i)
        plt.plot(X[img][1])
        plt.ylim([-0.1, 2.1])
        plt.title(img)
        i += 1

def plotResult(idx, result, truth):
    plt.figure(figsize=(12, 12))
    i = 0
    for img in idx:
        plt.subplot(331+i)
        plt.plot(result[img][0, :])
        plt.title(img)
        i += 1

    plt.figure(figsize=(12, 12))
    i = 0
    for img in idx:
        plt.subplot(331+i)
        plt.plot(truth[img])
        plt.title(img)
        i += 1
        
def plotDump(idx, X, plot_range):
    if plot_range == 'None':
        plot_range = [0, len(X[idx][0])]
    plt.figure(figsize=(12, 3))
    plt.subplot(121)
    plt.plot(X[idx][0])
    plt.xlim(plot_range)
    plt.title(['value, dump #', idx])
    plt.subplot(122)
    plt.plot(X[idx][1])
    plt.xlim(plot_range)
    plt.title(['label, dump #', idx])