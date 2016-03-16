import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import (
    pdist,
    squareform,
)
from sklearn.cross_validation import train_test_split
from sklearn.datasets import (
    make_circles,
    make_moons,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from visualization import plot_decision_regions


def plot_manual_lda_transformation(X, y):
    np.set_printoptions(precision=4)

    print("Class label distribution: %s" % np.bincount(y)[1:])

    num_features = 13
    mean_vectors = []
    for label in range(1, 4):
        mean_vectors.append(
            np.mean(X[y == label], axis=0).reshape(num_features, 1)
        )
        print("MV %s: %s\n" % (label, mean_vectors[label-1].T))
    mean_overall = np.mean(X, axis=0).reshape(num_features, 1)

    S_W_unscaled = np.zeros((num_features, num_features))
    S_W = np.zeros((num_features, num_features))
    S_B = np.zeros((num_features, num_features))
    for label, mean_vector in zip(range(1, 4), mean_vectors):
        class_scatter = np.zeros((num_features, num_features))
        for row in X[y == label]:
            row = row.reshape(num_features, 1)
            class_scatter += (row - mean_vector).dot((row - mean_vector).T)
        S_W_unscaled += class_scatter

        S_W += np.cov(X[y == label].T)

        n = X[y == label, :].shape[0]
        S_B += n * (mean_vector - mean_overall).dot(
            (mean_vector - mean_overall).T
        )

    print(
        "Unscaled within-class scatter matrix: %sx%s" %
        (S_W_unscaled.shape[0], S_W_unscaled.shape[1])
    )
    print(
        "Scaled within-class scatter matrix: %sx%s" %
        (S_W.shape[0], S_W.shape[1])
    )
    print(
        "Between-class scatter matrix: %sx%s" %
        (S_B.shape[0], S_B.shape[1])
    )

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eigenpairs = [
        (np.abs(eigenvalue), eigenvectors[:, index])
        for index, eigenvalue
        in enumerate(eigenvalues)
    ]
    eigenpairs = sorted(eigenpairs, key=lambda k: k[0], reverse=True)

    print("Eigenvalues in decreasing order: \n")
    for eigenpair in eigenpairs:
        print(eigenpair[0])

    tot = sum(eigenvalues.real)
    discr = [i/tot for i in map(lambda p: p[0], eigenpairs)]
    cum_discr = np.cumsum(discr)

    plt.bar(
        range(1, 14),
        discr,
        alpha=0.5,
        align='center',
        label='individual "discriminability"',
    )
    plt.step(
        range(1, 14),
        cum_discr,
        where='mid',
        label='cumulative "discriminability"',
    )
    plt.ylabel('"discriminability" ratio')
    plt.xlabel('Linear Discriminants')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='best')
    plt.show()

    w = np.hstack((
        eigenpairs[0][1][:, np.newaxis].real,
        eigenpairs[1][1][:, np.newaxis].real,
    ))
    print('Matrix W:\n', w)

    X_lda = X.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for label, color, marker in zip(np.unique(y), colors, markers):
        plt.scatter(
            X_lda[y == label, 0],
            X_lda[y == label, 1],
            c=color,
            label=label,
            marker=marker,
        )

    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='upper right')
    plt.show()


def plot_sklearn_lda_with_lr(X_train, X_test, y_train, y_test):
    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train, y_train)

    lr = LogisticRegression()
    lr = lr.fit(X_train_lda, y_train)

    plot_decision_regions(X_train_lda, y_train, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.show()

    X_test_lda = lda.transform(X_test)

    plot_decision_regions(X_test_lda, y_test, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.show()


def plot_manual_pca_transformation(X, y):
    cov_mat = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    print("\nEigenvalues \n%s" % eigenvalues)

    tot = sum(eigenvalues)
    var_exp = [i/tot for i in sorted(eigenvalues, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    plt.bar(
        range(1, 14),
        var_exp,
        alpha=0.5,
        align='center',
        label='individual explained variance',
    )
    plt.step(
        range(1, 14),
        cum_var_exp,
        where='mid',
        label='cumulative explained variance',
    )
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()

    eigenpairs = [
        (np.abs(eigenvalue), eigenvectors[:, index])
        for index, eigenvalue
        in enumerate(eigenvalues)
    ]
    eigenpairs.sort(reverse=True)

    w = np.hstack((
        eigenpairs[0][1][:, np.newaxis],
        eigenpairs[1][1][:, np.newaxis],
    ))
    print('Matrix W:\n%s\n' % w)

    X_pca = X.dot(w)

    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for label, color, marker in zip(np.unique(y), colors, markers):
        plt.scatter(
            X_pca[y == label, 0],
            X_pca[y == label, 1],
            c=color,
            label=label,
            marker=marker,
        )

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()

    print(X_pca[0])


def plot_sklearn_pca_with_lr(X_train, X_test, y_train, y_test):
    pca = PCA()
    pca.fit(X_train)
    print(pca.explained_variance_ratio_)

    plt.bar(
        range(1, 14),
        pca.explained_variance_ratio_,
        alpha=0.5,
        align='center',
    )
    plt.step(
        range(1, 14),
        np.cumsum(pca.explained_variance_ratio_),
        where='mid',
    )
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.show()

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()

    lr = LogisticRegression()
    lr = lr.fit(X_train_pca, y_train)

    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()

    plot_decision_regions(X_test_pca, y_test, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()


def get_standardized_wine_data():
    df = pd.read_csv(os.path.join('datasets', 'wine.data'), header=None)
    df.columns = [
        'Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
        'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
        'Proanthocyanins', 'Color intensity', 'Hue',
        'OD280/OD315 of diluted wines', 'Proline',
    ]
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
    )
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std, X_test_std, y_train, y_test


def rbf_kernel_pca(X, gamma, n_components):
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    K = exp(-gamma * mat_sq_dists)

    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    eigenvalues, eigenvectors = eigh(K)

    X_pc = np.column_stack((
        eigenvectors[:, -i] for i in range(1, n_components+1)
    ))

    return X_pc


def plot_pca_for_data(data_type, n_samples):
    if data_type == 'half_circles':
        X, y = make_moons(n_samples=n_samples, random_state=123)
        format_x_axis = True
    elif data_type == 'concentric_circles':
        X, y = make_circles(
            n_samples=n_samples,
            random_state=123,
            noise=0.1,
            factor=0.2,
        )
        format_x_axis = False

    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color='red',
        marker='^',
        alpha=0.5,
    )
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color='blue',
        marker='o',
        alpha=0.5,
    )
    plt.show()

    X_spca = PCA(n_components=2).fit_transform(X)
    X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

    for index, X_pca in enumerate((X_spca, X_kpca)):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

        ax[0].scatter(
            X_pca[y == 0, 0],
            X_pca[y == 0, 1],
            color='red',
            marker='^',
            alpha=0.5,
        )
        ax[0].scatter(
            X_pca[y == 1, 0],
            X_pca[y == 1, 1],
            color='blue',
            marker='o',
            alpha=0.5,
        )

        ax[1].scatter(
            X_pca[y == 0, 0],
            np.zeros((n_samples/2, 1))+0.02,
            color='red',
            marker='^',
            alpha=0.5,
        )
        ax[1].scatter(
            X_pca[y == 1, 0],
            np.zeros((n_samples/2, 1))-0.02,
            color='blue',
            marker='o',
            alpha=0.5,
        )

        ax[0].set_xlabel('PC1')
        ax[0].set_ylabel('PC2')
        ax[1].set_ylim([-1, 1])
        ax[1].set_yticks([])
        ax[1].set_xlabel('PC1')
        if format_x_axis and (index == 1):
            ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
            ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

        plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_standardized_wine_data()
    # plot_manual_pca_transformation(X_train, y_train)
    # plot_sklearn_pca_with_lr(X_train, X_test, y_train, y_test)
    # plot_manual_lda_transformation(X_train, y_train)
    # plot_sklearn_lda_with_lr(X_train, X_test, y_train, y_test)
    # plot_pca_for_data(data_type='half_circles', n_samples=100)
    plot_pca_for_data(data_type='concentric_circles', n_samples=1000)
