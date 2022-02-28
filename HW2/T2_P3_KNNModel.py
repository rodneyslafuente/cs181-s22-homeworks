import numpy as np
import pandas as pd

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    def __distance(self, x_i, x_j):
        return ((x_i[0] - x_j[0]) / 3) ** 2 + (x_i[1] - x_j[1]) ** 2

    def __k_nearest(self, X, x_i):
        N = len(X)
        distances = [(j, self.__distance(x_i, X[j])) for j in range(N)]
        sorted_distances = sorted(distances, key=lambda tup: tup[1])
        nearest_k = [X[tup[0]] for tup in sorted_distances[:self.K]]
        return nearest_k

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

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y

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

        self.fit(X, y)
        self.__k_nearest(X, X[0])




if __name__ == '__main__':
    model = KNNModel(3)
    model.test()
