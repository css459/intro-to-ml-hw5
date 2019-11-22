import numpy as np
from sklearn.metrics import accuracy_score

from src.pegasos import Pegasos


class OneVAllClassifier:

    def __init__(self, lamb=2 ** -5):
        self.lamb = lamb

        # Dictionary of {target_class: [vector]}
        self.weight_vectors = {}

        # Key in self.weight_vectors
        self.best = None

        # TODO
        self.clf = Pegasos(lamb=lamb)

    @staticmethod
    def _relabel(y, target_value):
        return np.where(y == target_value, 1, -1)

    def _fit_one_append(self, X, y, target_class):
        yy = OneVAllClassifier._relabel(y, target_class)
        self.weight_vectors[target_class] = self.clf.fit(X, yy)

    def fit(self, X, y):
        # Fit a classifier and get the weights
        # for each class
        classes = np.unique(y)
        for c in classes:
            self._fit_one_append(X, y, c)

    def predict(self, X):
        # Array of (prediction_proba, class)
        y = []

        # Get prediction for each class
        for x in X:
            preds = []
            for c in self.weight_vectors.keys():
                w = self.weight_vectors[c]
                preds.append((np.dot(x, w), c))

            # Return the maximum
            best = max(preds)[1]
            y.append(best)

        print(y)

        return y

    def test(self, X, y):
        p = self.predict(X)
        return accuracy_score(y, p)
