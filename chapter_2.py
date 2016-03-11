import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

from visualization import plot_decision_regions


class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


class AdalineSGD(object):

    def __init__(self, eta=0.01, n_iter=10, shuffle=True):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for x_i, target in zip(X, y):
                cost.append(self._update_weights(x_i, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            if len(X.shape) == 1:
                m = X.shape[0]
            else:
                m = X.shape[1]
            self._initialize_weights(m)
        if y.ravel().shape[0] > 1:
            for x_i, target in zip(X, y):
                self._update_weights(x_i, target)
        else:
            self._update_weights(X, y)
        return self

    @staticmethod
    def _shuffle(X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.w_ = np.zeros(1+m)
        self.w_initialized = True

    def _update_weights(self, x_i, target):
        output = self.net_input(x_i)
        error = target - output
        self.w_[1:] += self.eta * x_i.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


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
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_adalinegd_results(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ada1 = AdalineGD(eta=0.01, n_iter=10).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = AdalineGD(eta=0.0001, n_iter=10).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.show()


def plot_adalinesgd_different_online_results(X_std, y):
    ada_no_online = AdalineSGD(eta=0.01, n_iter=1).fit(X_std, y)
    plot_adalinesgd_results(
        X_std,
        y,
        ada_no_online,
        label='No Online',
        show_cost=False,
    )

    starting_indexes = np.random.permutation(len(y))[:50]
    ada_some_online = AdalineSGD(eta=0.01, n_iter=1).fit(
        X_std[starting_indexes],
        y[starting_indexes],
    )
    for index in range(len(y)):
        if index not in starting_indexes:
            x_i = X_std[index]
            target = y[index]
            ada_some_online = ada_some_online.partial_fit(x_i, target)
    plot_adalinesgd_results(
        X_std,
        y,
        ada_some_online,
        label='Some Online',
        show_cost=False,
    )

    ada_all_online = AdalineSGD(eta=0.01)
    for x_i, target in zip(X_std, y):
        ada_all_online = ada_all_online.partial_fit(x_i, target)
    plot_adalinesgd_results(
        X_std,
        y,
        ada_all_online,
        label='All Online',
        show_cost=False,
    )


def plot_adalinesgd_results(X_std, y, clf=None, label=None, show_cost=True):
    if clf is None:
        clf = AdalineSGD(eta=0.01, n_iter=15).fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=clf)
    title = "Adaline - Stochastic Gradient Descent%(label)s" % {
        'label': '' if label is None else (" - %s" % label),
    }
    plt.title(title)
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()

    if show_cost:
        plt.plot(range(1, len(clf.cost_) + 1), clf.cost_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Average Cost')
        plt.show()


def plot_adalinegd_standardized_results(X_std, y):
    ada = AdalineGD(eta=0.01, n_iter=15).fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')

    plt.show()


def plot_iris_data(X, y):
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
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()


def plot_perceptron_results(X, y):
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    iris_data = datasets.load_iris()
    y = iris_data['target'][0:100]
    y = np.where(y == 0, -1, 1)
    X = iris_data['data'][0:100][:, [0, 2]]

    plot_iris_data(X, y)

    # plot_perceptron_results(X, y)
    # plot_adalinegd_results(X, y)

    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # plot_adalinegd_standardized_results(X_std, y)

    np.random.seed(1)
    # plot_adalinesgd_results(X_std, y)

    plot_adalinesgd_different_online_results(X_std, y)
