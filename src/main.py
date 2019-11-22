from src.one_v_all import OneVAllClassifier
from src.preprocess import load_test, load_train

X, y = load_train()

# Best: (0.8674999999999999, 0.125)
# best_lambda = find_best(X, y)
# print("Best:", best_lambda)


# Retrain using best lambda
lamb = 0.125
best_ova = OneVAllClassifier(lamb=lamb)

# Get test error
X_test, y_test = load_test()
best_ova.fit(X, y)
accuracy = best_ova.test(X_test, y_test)

# Best Test Accuracy: 0.873
print("Best Test Accuracy:", accuracy)
