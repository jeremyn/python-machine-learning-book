import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import (
    LogisticRegression,
    Perceptron,
)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import datasets

from visualization import plot_decision_regions


def gini(p):
    return 2 * p * (1-p)


def entropy(p):
    return -p * np.log2(p) - (1-p) * np.log2(1-p)


def error(p):
    return 1 - max(p, 1-p)


def plot_impurity_indexes():
    probs = np.arange(0.0, 1.0, 0.01)
    entropies = [entropy(p) if p != 0 else None for p in probs]
    scaled_entropies = [e * 0.5 if e is not None else None for e in entropies]
    errors = [error(p) for p in probs]

    plt.figure()
    ax = plt.subplot(111)

    plots = (
        (entropies, 'Entropy', '-', 'black'),
        (scaled_entropies, 'Entropy (scaled)', '-', 'lightgray'),
        (gini(probs), 'Gini Impurity', '--', 'red'),
        (errors, 'Misclassification Error', '-.', 'green'),
    )

    for y, label, linestyle, color in plots:
        ax.plot(probs, y, label=label, linestyle=linestyle, lw=2, color=color)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        fancybox=True,
        shadow=False,
    )
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')

    plt.ylim([0, 1.1])
    plt.xlabel('p(i=1)')
    plt.ylabel('Impurity Index')

    plt.show()


def plot_iris_with_classifier(clf, print_accuracy=False, standardize=True):
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
    )

    if standardize:
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
        units = 'standardized'
    else:
        units = 'cm'

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    if print_accuracy:
        print("Misclassified samples: %d" % (y_test != y_pred).sum())
        print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(
        X=X_combined,
        y=y_combined,
        classifier=clf,
        test_index=range(105, 150),
    )

    plt.xlabel("petal length [%s]" % units)
    plt.ylabel("petal width [%s]" % units)
    plt.legend(loc='upper left')

    plt.show()


def plot_lr_regularization():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
    )

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)

    weights = []
    params = []
    for c in np.logspace(-5, 4, num=10):
        lr = LogisticRegression(C=c, random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(c)
    weights = np.array(weights)
    plt.plot(params, weights[:, 0], label='petal length')
    plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.show()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def plot_sigmoid():
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)

    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')

    plt.yticks([0.0, 0.5, 1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)

    plt.show()


def plot_xor():
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)

    svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
    svm.fit(X_xor, y_xor)
    plot_decision_regions(X_xor, y_xor, classifier=svm)

    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    # clf = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    # clf = LogisticRegression(C=1000.0, random_state=0)
    # clf = SVC(kernel='linear', C=1.0, random_state=0)
    # clf = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
    # clf = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
    # plot_iris_with_classifier(clf)

    # plot_sigmoid()
    # plot_lr_regularization()
    # plot_xor()
    plot_impurity_indexes()
