import numpy as np



# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __logSigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __softmax(self, z):
        denom = sum([np.exp(z_j) for z_j in z])
        probs = [np.exp(z_i) / denom for z_i in z]
        return probs

    def test(self):
        z = np.array([4, 1, 7])
        sm = self.__softmax(z)
        print(sm)

    def __gradient(self, X, y):

        pass


    # TODO: Implement this method!
    def fit(self, X, y):
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


if __name__ == "__main__":

    eta = 0.001
    lam = 2

    model = LogisticRegression(eta=eta, lam=lam)
    model.test()