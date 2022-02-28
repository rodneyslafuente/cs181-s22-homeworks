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

        # counts[k] = no. data points classified as C_k (scalar)
        counts = np.zeros(self.K)
        for i in range(N):
            counts[y[i]] += 1

        # Number of features in a data point (in our case, this is 2)
        p = X.shape[1]
        assert(p == 2)

        # Fit a list of different covariance matrices for each class. 
        # Sigma_list[k] = covariance matrix for class C_k
        self.Sigma_list = np.zeros((self.K, p, p))

        for k in range(self.K):
            Sigma_k = np.zeros((p, p))
            for i in range(N):
                Sigma_k += encoded_y[i][k] * np.dot((X[i] - self.mu[k]).reshape(-1, 1), (X[i] - self.mu[k]).reshape(-1, 1).T)
            Sigma_k /= counts[k]
            self.Sigma_list[k] = Sigma_k

        assert(self.Sigma_list.shape == (self.K, p, p))
        
        # Find the average of all covariance matrices for each class.
        # Set this average as the shared covariance matrix for all classes.
        self.Sigma = np.zeros((p, p))

        for k in range(self.K):
            self.Sigma += (counts[k] / N) * (self.Sigma_list[k])

        assert(self.Sigma.shape == (p, p))
 
    def fit(self, X, y):
        self.__fitPi(X, y)
        self.__fitMu(X, y)
        self.__fitSigma(X, y)

    def predict(self, X_pred):
        preds = []
        for x_i in X_pred:
            # probs_for_x[k] = probability that x is classified as C_k, times 
            # a scalar that is the same for all items in the list (we are only
            # concerned with picking the greatest value in this list, so we can 
            # ignore this scalar)
            probs_for_x = np.zeros(self.K)
            for k in range(self.K):
                # Here we pick whether we are using a shared or individual covariance
                # matrix for each class
                Sigma_to_use = self.Sigma_list[k]
                if self.is_shared_covariance:
                    Sigma_to_use = self.Sigma
                # Here we use Bayes' rule to make a prediction
                probs_for_x[k] = mvn.pdf(x_i, self.mu[k], Sigma_to_use) * self.pi[k]
            preds.append(np.argmax(probs_for_x))
        return np.array(preds)

    def negative_log_likelihood(self, X, y):
        N = len(X)
        log_like = 0
        encoded_y = np.array([self.__one_hot(y_i) for y_i in y])
        for i in range(N):
            for k in range(self.K):
                # Here we pick whether we are using a shared or individual covariance
                # matrix for each class
                Sigma_to_use = self.Sigma_list[k]
                if self.is_shared_covariance:
                    Sigma_to_use = self.Sigma
                if encoded_y[i][k] == 1:
                    log_like += np.log(mvn.pdf(X[i], self.mu[k], Sigma_to_use) * self.pi[k])
        return -log_like

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

        self.fit(X, y)
        y_hat = self.predict(X)
        loss = self.negative_log_likelihood(X, y_hat)
        print(loss)

        self.is_shared_covariance = True

        self.fit(X, y)
        y_hat = self.predict(X)
        loss = self.negative_log_likelihood(X, y_hat)
        print(loss)

if __name__ == '__main__':
    model = GaussianGenerativeModel()
    model.test()
