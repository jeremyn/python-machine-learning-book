# Copyright Jeremy Nation.
# Released under the MIT license. See included LICENSE.txt.
#
# Almost entirely copied from code created by Sebastian Raschka released under
# the MIT license. See included LICENSE.raschka.txt.
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_regions(X, y, classifier, resolution=0.02, test_index=None):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for index, class_ in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == class_, 0],
            y=X[y == class_, 1],
            alpha=0.8,
            c=cmap(index),
            marker=markers[index],
            label=class_,
        )

    if test_index is not None:
        X_test = X[test_index, :]
        plt.scatter(
            X_test[:, 0],
            X_test[:, 1],
            s=55,
            c='',
            marker='o',
            alpha=1.0,
            linewidths=1,
            label='test set',
        )
