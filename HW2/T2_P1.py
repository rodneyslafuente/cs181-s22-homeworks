import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import matplotlib.patches as mpatches
from scipy.special import expit as sigmoid

# This script requires the above packages to be installed.
# Please implement the basis2, basis3, fit, and predict methods.
# Then, create the three plots. An example has been included below, although
# the models will look funny until fit() and predict() are implemented!

# You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

def basis1(x):
    return np.stack([np.ones(len(x)), x], axis=1)

def basis2(x):
    return np.array([[1, x_i, x_i**2] for x_i in x])

def basis3(x):
    return np.array([[1, x_i, x_i**2, x_i**3, x_i**4, x_i**5] for x_i in x])

class LogisticRegressor:
    def __init__(self, eta, runs, N):
        self.eta = eta
        self.runs = runs
        self.N = N

    def __logSigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __computeGradient(self, x, y, W):
        gradient = np.zeros((x.shape[1], ))
        for i in range(N):
            x_i = x[i]
            y_i = y[i][0]
            y_i_hat = self.__logSigmoid(np.dot(W.T, x_i))
            gradient += (y_i_hat - y_i) * x_i
        gradient = gradient.reshape(-1, 1)
        ## Averging the gradient over the data points, as is 
        ## specified in Problem 1.1.
        gradient /= N
        return gradient

    def fit(self, x, y, w_init=None):
        # Keep this if case for the autograder
        if w_init is not None:
            self.W = w_init
        else:
            self.W = np.random.rand(x.shape[1], 1)        
        
        # optimize W using gradient decsent
        new_W = np.empty_like(self.W)
        for _ in range(self.runs):
            gradient = self.__computeGradient(x, y, self.W)
            new_W = self.W - (self.eta * gradient)
            self.W = new_W

    def predict(self, x):
        y_hat = [self.__logSigmoid(np.dot(self.W.T, x_i)) for x_i in x]
        return np.array(y_hat)

# Function to visualize prediction lines
# Takes as input last_x, last_y, [list of models], basis function, title,
# last_x and last_y should specifically be the dataset that the last model
# in [list of models] was trained on
def visualize_prediction_lines(last_x, last_y, models, basis, title):
    # Plot setup
    green = mpatches.Patch(color='green', label='Ground truth model')
    black = mpatches.Patch(color='black', label='Mean of learned models')
    purple = mpatches.Patch(color='purple', label='Model learned from displayed dataset')
    plt.legend(handles=[green, black, purple], loc='upper right')
    plt.title(title)
    plt.xlabel('X Value')
    plt.ylabel('Y Label')
    plt.axis([-3, 3, -.1, 1.1]) # Plot ranges

    # Plot dataset that last model in models (models[-1]) was trained on
    cmap = c.ListedColormap(['r', 'b'])
    plt.scatter(last_x, last_y, c=last_y, cmap=cmap, linewidths=1, edgecolors='black')

    # Plot models
    X_pred = np.linspace(-3, 3, 1000)
    X_pred_transformed = basis(X_pred)

    ## Ground truth model
    plt.plot(X_pred, np.sin(1.2*X_pred) * 0.4 + 0.5, 'g', linewidth=5)

    ## Individual learned logistic regressor models
    Y_hats = []
    for i in range(len(models)):
        model = models[i]
        Y_hat = model.predict(X_pred_transformed)
        Y_hats.append(Y_hat)
        if i < len(models) - 1:
            plt.plot(X_pred, Y_hat, linewidth=.3)
        else:
            plt.plot(X_pred, Y_hat, 'purple', linewidth=3)

    # Mean / expectation of learned models over all datasets
    plt.plot(X_pred, np.mean(Y_hats, axis=0), 'k', linewidth=5)

    plt.savefig(title + '.png')

# Function to generate datasets from underlying distribution
def generate_data(dataset_size):
    x, y = [], []
    for _ in range(dataset_size):
        x_i = 6 * np.random.random() - 3
        p_i = np.sin(1.2 * x_i) * 0.4 + 0.5
        y_i = np.random.binomial(1, p_i)
        x.append(x_i)
        y.append(y_i)
    return np.array(x), np.array(y).reshape(-1, 1)

if __name__ == "__main__":
    
    # DO NOT CHANGE THE SEED!
    np.random.seed(1738)
    eta = 0.001
    runs = 10000
    N = 30
    num_models = 10

    # Make plot for each basis with all models on each plot
    for basis in [basis1, basis2, basis3]:
        all_models = []
        for _ in range(num_models):
            x, y = generate_data(N)
            x_transformed = basis(x)
            model = LogisticRegressor(eta=eta, runs=runs, N=N)
            model.fit(x_transformed, y)
            all_models.append(model)
        plt.figure()
        visualize_prediction_lines(x, y, all_models, basis, basis.__name__)