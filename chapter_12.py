import os

import matplotlib.pyplot as plt
import numpy as np


def get_mnist_data():
    path = os.path.join('datasets', 'mnist')
    mnist_data = []
    for kind in ('train', 't10k'):
        labels_path = os.path.join(path, "%s-labels-idx1-ubyte" % kind)
        images_path = os.path.join(path, "%s-images-idx3-ubyte" % kind)

        with open(labels_path, 'rb') as lbpath:
            lbpath.seek(8)
            mnist_data.append(np.fromfile(lbpath, dtype=np.uint8))

        with open(images_path, 'rb') as imgpath:
            imgpath.seek(16)
            mnist_data.append(
                np.fromfile(
                    imgpath,
                    dtype=np.uint8,
                ).reshape(len(mnist_data[-1]), 784)
            )

    y_train, X_train, y_test, X_test = mnist_data
    print(
        "Train: rows: %d, columns: %d" % (X_train.shape[0], X_train.shape[1])
    )
    print("Test: rows: %d, columns: %d" % (X_test.shape[0], X_test.shape[1]))
    return X_train, X_test, y_train, y_test


def display_mnist_examples(X, y):
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X[y == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.show()

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(25):
        img = X[y == 7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.show()


if __name__ == '__main__':
    path = os.path.join('datasets', 'mnist')
    X_train, X_test, y_train, y_test = get_mnist_data()
    display_mnist_examples(X_train, y_train)
