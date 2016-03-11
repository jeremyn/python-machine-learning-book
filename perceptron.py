import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for x_i, target in zip(X, y):
                update = self.eta * (target - self.predict(x_i))
                self.w_[1:] += update * x_i
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


if __name__ == '__main__':
    iris_data = datasets.load_iris()
    y = iris_data['target'][0:100]
    y = np.where(y == 0, -1, 1)
    X = iris_data['data'][:, [0, 2]]
    plt.scatter(
        X[:50, 0],
        X[:50, 1],
        color='red',
        marker='o',
        label='setosa',
    )
    plt.scatter(
        X[50:100, 0],
        X[50:100, 1],
        color='blue',
        marker='x',
        label='versicolor',
    )
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()