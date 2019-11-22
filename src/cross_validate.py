import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from src.one_v_all import OneVAllClassifier


def cv(X, y, lamb):
    print("Cross Validating Lambda:", lamb)
    n = 5
    acc = 0
    kf = KFold(n_splits=n)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ova = OneVAllClassifier(lamb=lamb)
        ova.fit(X_train, y_train)

        # Add the accuracy score to the accumulator
        acc += ova.test(X_test, y_test)

    # Return average accuracy
    return acc / n


def plot_lambdas(cv_score, lambdas):
    plt.plot(lambdas, cv_score)
    plt.title("Cross Validated Avg. Accuracy vs. Lambda \nfor One Vs All Pegasos")
    plt.xlabel("Lambda for Pegasos (log2)")
    plt.ylabel("CV Average Accuracy (k = 5)")
    plt.show()


def find_best(X, y):
    lambas = [2 ** i for i in range(-5, 2)]
    cv_score = [cv(X, y, lamb=l) for l in lambas]

    # Plot the scores
    plot_lambdas(cv_score, lambas)

    return sorted(zip(cv_score, lambas), reverse=True)[0]
