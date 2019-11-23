from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from src.preprocess import load_test, load_train


def make_default_classifier(X_train, y_train, X_test, y_test):
    clf = SVC()
    ovr = OneVsRestClassifier(estimator=clf, n_jobs=-1)

    ovr.fit(X_train, y_train)

    # Evaluate
    acc = accuracy_score(y_test, ovr.predict(X_test))
    print(acc)

    return ovr


def make_opt_classifier(X_train, y_train, X_test, y_test):
    # Do vanilla CV on default values
    n = 10
    acc = 0
    kf = KFold(n_splits=n)
    for train_index, test_index in kf.split(X):
        X_tr, X_te = X_train[train_index], X_train[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]

        ova = OneVsRestClassifier(estimator=SVC(), n_jobs=-1)
        ova.fit(X_tr, y_tr)

        # Add the accuracy score to the accumulator
        acc += accuracy_score(y_te, ova.predict(X_te))

    # Return average accuracy
    avg_cv_acc = acc / n

    print("VANILLA CV SCORE:", avg_cv_acc)

    # Find BEST parameters using a grid search with 10 Folds
    # By default  for this training set, SKLearn will use
    # gamma = 1 / (n_features * X.var()) = 0.003353
    params = {
        'C': [0.1, 0.5, 1.0, 1.5, 2.0],
        'gamma': ['auto', 'scale', 0.007, 0.01, 0.001]
    }
    grid = GridSearchCV(estimator=SVC(), param_grid=params, n_jobs=-1, cv=n, verbose=2)
    grid.fit(X_train, y_train)

    print(grid.best_estimator_)
    print(grid.best_params_)
    print("OPTIMUM SCORE:", grid.best_score_)

    return grid


def make_best_classifier(X_train, y_train, X_test, y_test, C=2.0, gamma=0.007):
    ovr = OneVsRestClassifier(estimator=SVC(C=C, gamma=gamma), n_jobs=-1)
    ovr.fit(X_train, y_train)

    # Evaluate
    print("BEST TEST SCORE:", accuracy_score(y_test, ovr.predict(X_test)))

    return ovr


if __name__ == '__main__':
    X, y = load_train()
    X_t, y_t = load_test()

    # Problem D: Train a default OneVRest
    # Classifier with default RBF SVC
    # Evaluate on test set
    # ACCURACY: 0.91
    # ovr1 = make_default_classifier(X, y, X_t, y_t)

    # Problem E, F: Find best hyper params and
    # vanilla CV score for default SVC
    #
    # VANILLA CV SCORE 10 FOLD: 0.9055
    # {'C': 2.0, 'gamma': 0.007}
    # OPTIMUM CV FOLD 10 SCORE: 0.949
    # grid = make_opt_classifier(X, y, X_t, y_t)

    # Test the best found parameters on test data
    ovr_best = make_best_classifier(X, y, X_t, y_t)
