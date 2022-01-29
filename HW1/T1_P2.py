#####################
# CS 181, Spring 2022
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# set up data
data = [(0., 0.),
        (1., 0.5),
        (2., 1),
        (3., 2),
        (4., 1),
        (6., 1.5),
        (8., 0.5)]

x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])

x_test = np.arange(0, 12, .1)


def compute_kernel(x_1, x_2, tau):
    return math.exp((- (x_2 - x_1) ** 2) / tau)

def find_k_nearest(x, k, tau):
    return sorted(data, key=lambda tup: compute_kernel(tup[0], x, tau))[-k:]

def predict_y(x, k, tau):
    return (1 / k) * sum(list(map(lambda tup: tup[1], find_k_nearest(x, k, tau))))
    
def predict_knn(k=1, tau=1):
    """Returns predictions for the values in x_test, using KNN predictor with the specified k."""
    return list(map(lambda x: predict_y(x, k, tau), x_test))

# plot predicitions
plt.figure()
plt.scatter(x_train, y_train, label="training data", color='black')

def plot_knn_preds(k):
    y_test = predict_knn(k=k)
    plt.plot(x_test, y_test, label="predictions using k = "+str(k))

for k in (1, 3, len(x_train)-1):
    plot_knn_preds(k)

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title("KNN Predictions")
plt.legend()
plt.savefig('P2.png')