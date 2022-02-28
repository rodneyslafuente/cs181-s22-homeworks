import numpy as np
import pandas as pd

# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam, K=3, runs=10):
        self.eta = eta
        self.lam = lam
        self.K = K
        self.runs = runs

    def __softmax(self, z):
        denom = sum([np.exp(z_j) for z_j in z])
        probs = [np.exp(z_i) / denom for z_i in z]
        return np.array(probs).reshape(-1, 1)

    def __add_bias(self, X):
        pass

    def __one_hot(self, y):
        encoding = np.zeros(self.K)
        encoding[y] = 1
        return encoding.reshape(-1, 1)

    def __gradient(self, X, y, W):
        N = len(X)
        grad = []

        for j in range(self.K):
            grad_j = np.zeros_like(X[0])
            for i in range(N):
                y_ij_hat = self.__softmax(np.dot(W.T, X[i]))[j]
                y_ij = y[i][j]
                grad_j += (y_ij_hat - y_ij) * X[i]
            grad.append(grad_j)

        # Ed Post #254: we adopt the convention that 
        # the derivative of a scalar function with respect
        # to a matrix/vector has the same dimensions and orientation
        # as said matrix/vector (hence the transpose)
        return np.array(grad).T

    def fit(self, X, y):
        # Assumes y is not encoded as one-hot vectors
        encoded_y = np.array([self.__one_hot(y_i) for y_i in y])

        # Initializes weight matrix
        self.W = np.random.rand(X.shape[1], self.K)  
        new_W = np.empty_like(self.W)

        # Optimize weights
        for _ in range(self.runs):
            gradient = self.__gradient(X, encoded_y, self.W)
            new_W = self.W - (self.eta * gradient)
            self.W = new_W

    def predict(self, X_pred):
        preds = []
        for x_i in X_pred:
            y_i_hat = self.__softmax(np.dot(self.W.T, x_i))
            preds.append(np.argmax(y_i_hat)) 
        return np.array(preds)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
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

        # one_hot
        self.K = 3
        y_i = 1
        encoded_y_i = self.__one_hot(y_i)
        assert(np.array_equal(encoded_y_i, np.array([[0], [1], [0]])))
    
        # softmax
        z = np.array([4, 1, 7])
        sm = self.__softmax(z)
        assert(np.array_equal(np.around(sm, 3), np.array([[0.047], [0.002], [0.950]])))

        # gradient (runs)
        encoded_y = np.array([self.__one_hot(y_i) for y_i in y])
        self.W = np.random.rand(X.shape[1], self.K)        
        grad = self.__gradient(X, encoded_y, self.W)

        # fit (runs)
        self.fit(X, y)

        # predict (runs)
        y_hat = self.predict(X)




if __name__ == "__main__":
    eta = 0.001
    lam = 2
    K = 3

    model = LogisticRegression(eta=eta, lam=lam, K=K)
    model.test()
