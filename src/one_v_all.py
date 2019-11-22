import numpy as np

from src.pegasos import Pegasos


class OneVAllClassifier:

    def __init__(self, lamb=2 ** -5):
        self.lamb = lamb

        # Dictionary of {target_class: [vector]}
        self.weight_vectors = {}

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
        preds = []

        # Get prediction for each class
        for c in self.weight_vectors.keys():
            w = self.weight_vectors[c]
            preds.append((c, np.dot(X, w)))

        print(preds)

        # Return the maximum
        return sorted(preds)[0][1]
