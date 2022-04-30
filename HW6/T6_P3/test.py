import numpy as np
import numpy.random as npr

X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900

# Q = np.arange(2 * X_SCREEN // X_BINSIZE * Y_SCREEN // Y_BINSIZE)
# Q = np.reshape(Q, (2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

# state = (1, 1)
# rel_x, rel_y = state

# print(np.argmax(Q[:, rel_x, rel_y]))

print(Q.shape)