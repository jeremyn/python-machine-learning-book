import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import (
    cross_val_score,
    StratifiedKFold,
    train_test_split,
)
from sklearn.decomposition import PCA
from sklearn.learning_curve import (
    learning_curve,
    validation_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)


def get_wdbc_data():
    df = pd.read_csv(os.path.join('datasets', 'wdbc.data'), header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1,
    )
    return X_train, X_test, y_train, y_test


def plot_learning_curve(X_train, y_train):
    pipe_lr = Pipeline([
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(penalty='l2', random_state=0)),
    ])

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=pipe_lr,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=10,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(
        train_sizes,
        train_mean,
        color='blue',
        marker='o',
        markersize=5,
        label='training accuracy',
    )
    plt.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color='blue',
    )

    plt.plot(
        train_sizes,
        test_mean,
        color='green',
        linestyle='--',
        marker='s',
        markersize=5,
        label='validation accuracy',
    )
    plt.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color='green',
    )

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])

    plt.show()


def plot_validation_curve(X_train, y_train):
    pipe_lr = Pipeline([
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(penalty='l2', random_state=0)),
    ])

    param_range = np.logspace(-3, 2, num=6)
    train_scores, test_scores = validation_curve(
        estimator=pipe_lr,
        X=X_train,
        y=y_train,
        param_name='clf__C',
        param_range=param_range,
        cv=10,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(
        param_range,
        train_mean,
        color='blue',
        marker='o',
        markersize=5,
        label='training accuracy',
    )
    plt.fill_between(
        param_range,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color='blue',
    )

    plt.plot(
        param_range,
        test_mean,
        color='green',
        linestyle='--',
        marker='s',
        markersize=5,
        label='validation accuracy',
    )
    plt.fill_between(
        param_range,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color='green',
    )

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1.0])

    plt.show()


def use_kfold_cross_validation(X_train, X_test, y_train, y_test):
    pipe_lr = Pipeline([
        ('scl', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('clf', LogisticRegression(random_state=1)),
    ])
    pipe_lr.fit(X_train, y_train)
    print("Test accuracy: %.3f\n" % pipe_lr.score(X_test, y_test))

    kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print(
            "Fold: %s, Class dist.: %s, Acc: %.3f" %
            (k+1, np.bincount(y_train[train]), score)
        )
    print(
        "\nCustom CV accuracy: %.3f +/- %.3f\n" %
        (np.mean(scores), np.std(scores)),
    )

    scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10)
    print("cross_val_score CV accuracy scores: %s" % scores)
    print(
        "cross_val_score CV accuracy: %.3f +/- %.3f" %
        (np.mean(scores), np.std(scores))
    )


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_wdbc_data()
    # use_kfold_cross_validation(X_train, X_test, y_train, y_test)
    # plot_learning_curve(X_train, y_train)
    plot_validation_curve(X_train, y_train)
