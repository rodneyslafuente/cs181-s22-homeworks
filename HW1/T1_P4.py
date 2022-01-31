#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt
import math

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20
    
    # Make bases
    X = []

    # \phi(x) = x^j for j = 1, ..., 5
    if part == 'a':
        for x in xx:
            data_point = [1]
            for j in range(1, 6):
                data_point.append(x ** j)
            X.append(data_point)

    # \phi(x) = exp(-(x - \mu)^2/25) for \mu = 1960, 1965, 1970, ..., 2010
    elif part == 'b':
        for x in xx:
            data_point = [1]
            for mu in range(1960, 2011, 5):
                data_point.append(math.exp((-(x - mu) ** 2)) / 25)
            X.append(data_point)

    # \phi(x) = cos(x / j) for j = 1, ..., 5
    elif part == 'c':
        for x in xx:
            data_point = [1]
            for j in range(1, 6):
                data_point.append(math.cos(x / j))
            X.append(data_point)

     # \phi(x) = cos(x / j) for j = 1, ..., 25
    elif part == 'd':
        for x in xx:
            data_point = [1]
            for j in range(1, 26):
                data_point.append(math.cos(x / j))
            X.append(data_point)

    return np.array(X)

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)

def RSS(Y, Yhat):
    res = 0
    for y, y_hat in zip(Y, Yhat):
        res += (y - y_hat) ** 2
    return res


plt.plot(years, republican_counts, 'o', label='Data')

# Plot and report sum of squared error for each basis
parts = ['a', 'b', 'c', 'd']
for part in parts:
    X = make_basis(years, part=part)
    w = find_weights(X, Y)
    grid_X = make_basis(grid_years, part=part)
    grid_Yhat  = np.dot(grid_X, w)
    Yhat = np.dot(X, w)
    rss = RSS(Y, Yhat)
    plt.plot(grid_years, grid_Yhat, '-', label='('+part+f') RSS: {rss: .2f}')

plt.legend()
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.savefig('P4_1.png')