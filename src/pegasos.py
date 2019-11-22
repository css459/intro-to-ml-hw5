import numpy as np
from sklearn.metrics import accuracy_score


class Pegasos:

    def __init__(self, lamb=2 ** -5, epochs=20):
        self.lamb = lamb
        self.epochs = epochs
        self.w = []

    def _objective(self, X, y, w_t, reg=True):
        """
        Objective function for Pegasos SVM.
        :param X:     Training design matrix
        :param y:     Training labels (-1 or 1)
        :param w_t:         Current weight vector
        :param reg:         Include regularization
        :return: Value for objective function.
        """
        acc = 0.0
        for i in range(len(X)):
            acc += max(0, 1 - (y[i] * (w_t @ X[i])))

        accuracy = (1 / len(X)) * acc
        if not reg:
            return accuracy

        confidence = (self.lamb / 2) * (np.linalg.norm(w_t) ** 2)
        return confidence + accuracy

    def fit(self, X, y):
        """
        Train a Pegasos SVM and return the weight vectors for the given
        regularization, epochs, and training data.
        :param X:     Training design matrix
        :param y:     Training labels (-1 or 1)
        :return: The weight vectors with equal dimensionality to
                 the dimension of the samples in X.
        """
        w = np.zeros((len(X[0])))
        t = 0

        train_hist = []

        for i in range(self.epochs):
            for j in range(len(X)):
                t += 1
                eta_t = 1 / (t * self.lamb)

                y_j = y[j]
                x_j = X[j]

                if y_j * (np.dot(w, x_j)) < 1:
                    w = (1 - eta_t * self.lamb) * w + (eta_t * y_j * x_j)
                else:
                    w = (1 - eta_t * self.lamb) * w

            # Evaluate the epoch
            obj = Pegasos._objective(X, y, w, self.lamb)
            train_hist.append([obj, t])
            print("[Epoch=" + str(i) + " | Iter=" + str(t) + "] Objective: " + str(obj))

        self.w = w
        return w

    def test(self, X, y):
        """
        Classifies the points in the provided test set
        with the provided Pegasos weights. Prints
        performance metrics for the classification.
        Returns the accuracy measure.
        :param X:  The test design matrix
        :param y:  The true test labels
        :return: Accuracy measure for test set
        """
        y_pred = np.sign(X @ self.w)

        # The SkLearn metrics cannot handle if a label is 0,
        # for scoring, assume a 0 is a 1.
        y_pred = np.where(y_pred == 0, 1, y_pred)

        # print("F1 Score:", f1_score(y, y_pred))
        # print("Accuracy:", accuracy_score(y, y_pred))
        # print(confusion_matrix(y, y_pred))

        return accuracy_score(y, y_pred)
