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
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import (
    learning_curve,
    validation_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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


def work_with_grid_search(X_train, X_test, y_train, y_test):
    pipe_svc = Pipeline([
        ('scl', StandardScaler()),
        ('clf', SVC(random_state=1)),
    ])

    param_range = np.logspace(-4, 3, num=8)
    param_grid = [
        {
            'clf__C': param_range,
            'clf__kernel': ['linear'],
        },
        {
            'clf__C': param_range,
            'clf__gamma': param_range,
            'clf__kernel': ['rbf'],
        },
    ]

    gs = GridSearchCV(
        estimator=pipe_svc,
        param_grid=param_grid,
        scoring='accuracy',
        cv=10,
    )
    gs = gs.fit(X_train, y_train)
    print("gs.best_score_: %s" % gs.best_score_)
    print("gs.best_params_: %s" % gs.best_params_)

    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    print("Test accuracy: %.3f" % clf.score(X_test, y_test))

    gs = GridSearchCV(
        estimator=pipe_svc,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
    )
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print(
        "\nSVM nested cross-validation accuracy: %.3f +/- %.3f" %
        (np.mean(scores), np.std(scores)),
    )

    gs = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=0),
        param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
        scoring='accuracy',
        cv=5,
    )
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print(
        "\nDecision tree nested cross-validation accuracy: %.3f +/- %.3f" %
        (np.mean(scores), np.std(scores)),
    )


def work_with_metrics(X_train, X_test, y_train, y_test):
    pipe_svc = Pipeline([
        ('scl', StandardScaler()),
        ('clf', SVC(random_state=1)),
    ])
    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('default confusion matrix:\n', confmat)
    print(
        'reversed confusion matrix:\n',
        confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[1, 0]),
    )

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')

    plt.show()

    print("Precision: %.3f" % precision_score(y_true=y_test, y_pred=y_pred))
    print("Recall: %.3f" % recall_score(y_true=y_test, y_pred=y_pred))
    print("F1: %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

    scorer = make_scorer(f1_score, pos_label=0)
    c_gamma_range = np.logspace(-2, 1, num=4)
    param_grid = [
        {
            'clf__C': c_gamma_range,
            'clf__kernel': ['linear'],
        },
        {
            'clf__C': c_gamma_range,
            'clf__gamma': c_gamma_range,
            'clf__kernel': ['rbf'],
        },
    ]

    gs = GridSearchCV(
        estimator=pipe_svc,
        param_grid=param_grid,
        scoring=scorer,
        cv=10,
    )
    gs.fit(X_train, y_train)
    print("gs.best_score_: %s" % gs.best_score_)
    print("gs.best_params_: %s" % gs.best_params_)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_wdbc_data()
    # use_kfold_cross_validation(X_train, X_test, y_train, y_test)
    # plot_learning_curve(X_train, y_train)
    # plot_validation_curve(X_train, y_train)
    # work_with_grid_search(X_train, X_test, y_train, y_test)
    work_with_metrics(X_train, X_test, y_train, y_test)
