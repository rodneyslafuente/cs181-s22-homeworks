#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import matplotlib.pyplot as plt
import numpy as np
import math

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

def compute_loss(tau):
    loss = 0

    for x_i, y_i in data:
        f_i = 0

        for x_n, y_n in data:
            # do not include K(x_n, x)y_n in sum when calculating f(x)
            # since we are calculating loss over training data
            if not x_n == x_i:
                f_i += math.exp( (- (x_n - x_i) ** 2) / tau) * y_n

        loss += (y_i - f_i) ** 2

    return loss

for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))


# Make plots for problem 1.4

def f(x, tau):
    res = 0

    for x_n, y_n in data:
        res += math.exp( (- (x_n - x) ** 2) / tau) * y_n
    
    return res

x = np.arange(0, 12, 0.1)

plt.rcParams.update({ "text.usetex": True })
plt.figure()

for tau in (0.01, 2, 100):
    plt.plot(x, [f(x_i, tau) for x_i in x], label=r'$\tau = $'+str(tau), linestyle='', marker='o')

plt.xlabel('$x^*$')
plt.ylabel('$f(x^*)$')
plt.title('$f(x^*)$ calculated with different lengthscales')
plt.legend();

plt.savefig('P1.png');