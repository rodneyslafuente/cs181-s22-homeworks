#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

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
            # Calculate f(x_i) without including K(x_i, x_i) in sum
            if not x_n == x_i:
                f_i += math.exp( (- (x_n - x_i) ** 2) / tau) * y_n

        loss += (y_i - f_i) ** 2

    return loss


for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))

