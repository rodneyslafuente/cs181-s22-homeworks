import numpy as np

x = np.array([1, 3, 5])

def norm(x):
    return x / sum(x)

print(norm(x))
x /= 1000
print(x, norm(x))