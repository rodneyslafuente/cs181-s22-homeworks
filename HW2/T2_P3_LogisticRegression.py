import numpy as np
import pandas as pd




# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam, K):
        self.eta = eta
        self.lam = lam
        self.K = K

    def __softmax(self, z):
        denom = sum([np.exp(z_j) for z_j in z])
        probs = [np.exp(z_i) / denom for z_i in z]
        return np.array(probs)

    def __gradient(self, X, y, W):
        # gradient = np.zeros((x.shape[1], ))

        # for i in range(len(x)):
        #     x_i = x[i]
        #     y_i = y[i][0]
        #     y_i_hat = self.__logSigmoid(np.dot(W.T, x_i))
        #     gradient += (y_i_hat - y_i) * x_i
        # gradient = gradient.reshape(-1, 1)

        # return gradient
        N = len(X)
        grad = 0

        for j in range(self.K):
            grad_j = 0
            for i in range(N):
                
                grad_j += 

        pass

    # TODO: Implement this method!
    def fit(self, X, y, W):
        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for x in X_pred:
            z = np.cos(x ** 2).sum()
            preds.append(1 + np.sign(z) * (np.abs(z) > 0.3))
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

        # softmax
        z = np.array([4, 1, 7])
        sm = self.__softmax(z)
        assert(np.array_equal(np.around(sm, 3), np.array([0.047, 0.002, 0.950])))

        # gradient
        self.W = np.random.rand(X.shape[1], 1)        
        self.__gradient(X, y, self.W)



if __name__ == "__main__":
    eta = 0.001
    lam = 2
    K = 3

    model = LogisticRegression(eta=eta, lam=lam, K=K)
    model.test()
