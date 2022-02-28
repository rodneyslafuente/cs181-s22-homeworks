from select import kevent
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False, K=3):
        self.is_shared_covariance = is_shared_covariance
        self.K = K

    def __one_hot(self, y):
        encoding = np.zeros(self.K)
        encoding[y] = 1
        return encoding.reshape(-1, 1)

    def __fitPi(self, X, y):
        N = len(X)

        # counts[k] = no. data points classified as C_k (scalar)
        counts = np.zeros(self.K)
        for i in range(N):
            counts[y[i]] += 1
        
        self.pi = np.array([counts[k] / N for k in range(self.K)])

    def __fitMu(self, X, y):
        N = len(X)

        # counts[k] = no. data points classified as C_k (scalar)
        # sums[k] = sum of x_i's classified as C_k (vector with same shape as x_i)
        counts = np.zeros(self.K)
        sums = np.zeros((self.K, X.shape[1]))
        for i in range(N):
            counts[y[i]] += 1
            sums[y[i]] += X[i]

        self.mu = [sums[k] / counts[k] for k in range(self.K)]

    def __fitSigma(self, X, y):
        N = len(X)
        encoded_y = np.array([self.__one_hot(y_i) for y_i in y])
        self.Sigma = np.zeros((X.shape[1], X.shape[1]))

        for i in range(N):
            for k in range(self.K):
                term_to_add = encoded_y[i][k] * np.dot((X[i] - self.mu[k]).reshape(-1, 1), (X[i] - self.mu[k]).reshape(-1, 1).T)
                assert(term_to_add.shape == self.Sigma.shape)
                self.Sigma += term_to_add
        
        self.Sigma /= N
    # TODO: Implement this method!
    def fit(self, X, y):
        encoded_y = np.array([self.__one_hot(y_i) for y_i in y])

        self.__fitPi(X, y)
        self.__fitMu(X, y)
        self.__fitSigma(X, y)
        return 

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for x in X_pred:
            z = np.sin(x ** 2).sum()
            preds.append(1 + np.sign(z) * (np.abs(z) > 0.3))
        return np.array(preds)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        pass

    def test(self):
        # A mapping from string name to id
        star_labels = {
            'Dwarf': 0,       # also corresponds to 'red' in the graphs
            'Giant': 1,       # also corresponds to 'blue' in the graphs
            'Supergiant': 2   # also corresponds to 'green' in the graphs
        }

        # Read from file and extract X and y
        df = pd.read_csv('data/hr.csv')
        X = df[['Magnitude', 'Temperature']].values
        y = np.array([star_labels[x] for x in df['Type']])

        self.__fitPi(X, y)
        self.__fitMu(X, y)
        self.__fitSigma(X, y)

if __name__ == '__main__':
    model = GaussianGenerativeModel()
    model.test()
