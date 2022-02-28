import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam, K=3, runs=50):
        self.eta = eta
        self.lam = lam
        self.K = K
        self.runs = runs

    def __softmax(self, z):
        denom = sum([np.exp(z_j) for z_j in z])
        probs = [np.exp(z_i) / denom for z_i in z]
        return np.array(probs).reshape(-1, 1)

    def __add_bias(self, X):        
        return np.insert(X, 0, 1, axis=1)

    def __one_hot(self, y):
        encoding = np.zeros(self.K)
        encoding[y] = 1
        return encoding.reshape(-1, 1)

    def __gradient(self, X, y, W):
        N = len(X)
        grad = []

        for j in range(self.K):
            grad_j = np.zeros_like(X[0])

            # add gradient of cross entropy loss
            for i in range(N):
                y_ij_hat = self.__softmax(np.dot(W.T, X[i]))[j]
                y_ij = y[i][j]
                grad_j += (y_ij_hat - y_ij) * X[i]

            # add gradient of L2 regularization term
            grad_j += self.lam * W.T[j]
            grad.append(grad_j)

        # Ed Post #254: we adopt the convention that 
        # the derivative of a scalar function with respect
        # to a matrix/vector has the same dimensions and orientation
        # as said matrix/vector (hence the transpose)
        return np.array(grad).T

    def __negLogLikLoss(self, X, y, W):
        N = len(X)
        loss = 0
        for i in range(N):
            for j in range(self.K):
                y_ij_hat = self.__softmax(np.dot(W.T, X[i]))[j]
                y_ij = y[i][j]
                loss -= y_ij * np.log(y_ij_hat)
        return loss

    def fit(self, X, y, track_loss=True):
        # Encode y as one-hot vectors
        encoded_y = np.array([self.__one_hot(y_i) for y_i in y])

        # Add bias variable to design matrix
        X_with_bias = self.__add_bias(X)

        # Initializes weight matrix
        self.W = np.random.rand(X_with_bias.shape[1], self.K)  
        new_W = np.empty_like(self.W)

        # array to track loss
        self.losses = np.zeros(self.runs)

        # Optimize weights
        for i in range(self.runs):
            gradient = self.__gradient(X_with_bias, encoded_y, self.W)
            new_W = self.W - (self.eta * gradient)
            if track_loss:
                self.losses[i] = self.__negLogLikLoss(X_with_bias, encoded_y, self.W)
            self.W = new_W

    def predict(self, X_pred):
        preds = []
        X_pred_with_bias = self.__add_bias(X_pred)
        for x_i in X_pred_with_bias:
            y_i_hat = self.__softmax(np.dot(self.W.T, x_i))
            preds.append(np.argmax(y_i_hat)) 
        return np.array(preds)

    def visualize_loss(self, output_file, show_charts=False):
        plt.figure()
        plt.title('Logistic Regression Loss with eta='+str(self.eta)+' and lambda='+str(self.lam))
        plt.xlabel('Number of Iterations')
        plt.ylabel('Negative Log-Likelihood Loss')
        plt.plot(np.arange(self.runs), self.losses, '-o')
        plt.savefig(output_file)
        if show_charts:
            plt.show()

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
        self.__gradient(X, encoded_y, self.W)

        # fit (runs)
        self.fit(X, y)

        # predict (runs)
        y_hat = self.predict(X)
        print('Actual y:\n', y, '\nPredicted y:\n', y_hat)

        # plot losses
        output_file = 'logistic_losses.png'
        self.visualize_loss(output_file=output_file, show_charts=True)

if __name__ == "__main__":
    eta = 0.001
    lam = 2
    K = 3

    model = LogisticRegression(eta=eta, lam=lam, K=K)
    model.test()
