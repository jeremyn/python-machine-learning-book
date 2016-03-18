from itertools import product
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import comb
from sklearn import datasets
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    clone,
)
from sklearn.cross_validation import (
    cross_val_score,
    train_test_split,
)
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import six
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import (
    _name_estimators,
    Pipeline,
)
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier


def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probs = [
        comb(n_classifier, k) * error**k * (1-error)**(n_classifier-k)
        for k in range(k_start, n_classifier+1)
    ]
    return sum(probs)


def plot_ensemble_error():
    error_range = np.arange(0.0, 1.01, 0.01)
    ensemble_errors = [
        ensemble_error(n_classifier=11, error=error) for error in error_range
    ]

    plt.plot(
        error_range,
        ensemble_errors,
        label='Ensemble error',
        linewidth=2,
    )
    plt.plot(
        error_range,
        error_range,
        linestyle='--',
        label='Base error',
        linewidth=2,
    )
    plt.xlabel('Base error')
    plt.ylabel('Base/Ensemble error')
    plt.legend(loc='upper left')
    plt.grid()

    plt.show()


def use_bagging_classifier():
    df = pd.read_csv(os.path.join('datasets', 'wine.data'), header=None)
    df.columns = [
        'Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
        'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
        'Proanthocyanins', 'Color intensity', 'Hue',
        'OD280/OD315 of diluted wines', 'Proline',
    ]
    df = df[df['Class label'] != 1]
    X = df[['Alcohol', 'Hue']].values
    y = df['Class label'].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=1,
    )

    tree = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=None,
        random_state=3,
    )
    bag = BaggingClassifier(
        base_estimator=tree,
        n_estimators=500,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        random_state=1
    )

    clfs = [tree, bag]
    labels = ['Decision tree', 'Bagging']

    for clf, label in zip(clfs, labels):
        clf = clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        clf_train = accuracy_score(y_train, y_train_pred)
        clf_test = accuracy_score(y_test, y_test_pred)
        print(
            "%s train/test accuracies %.3f/%.3f" %
            (label, clf_train, clf_test)
        )

    x_min = X_train[:, 0].min() - 1
    x_max = X_train[:, 0].max() + 1
    y_min = X_train[:, 1].min() - 1
    y_max = X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1),
    )

    f, axarr = plt.subplots(
        nrows=1,
        ncols=2,
        sharex='col',
        sharey='row',
        figsize=(8, 3),
    )

    for index, clf, tt in zip([0, 1], clfs, labels):
        clf.fit(X_train, y_train)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[index].contourf(xx, yy, Z, alpha=0.3)
        axarr[index].scatter(
            X_train[y_train == 0, 0],
            X_train[y_train == 0, 1],
            c='blue',
            marker='^',
        )
        axarr[index].scatter(
            X_train[y_train == 1, 0],
            X_train[y_train == 1, 1],
            c='red',
            marker='o',
        )
        axarr[index].set_title(tt)

    axarr[0].set_ylabel('Alcohol', fontsize=12)
    plt.text(9.8, -1, s='Hue', ha='center', va='center', fontsize=12)

    plt.show()


def use_majority_vote_classifier():
    iris = datasets.load_iris()
    X = iris.data[50:, [1, 2]]
    y = iris.target[50:]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=1,
    )

    clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
    clf2 = DecisionTreeClassifier(
        max_depth=1,
        criterion='entropy',
        random_state=0,
    )
    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

    pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

    mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

    all_clf = [pipe1, clf2, pipe3, mv_clf]
    clf_labels = [
        'Logistic Regression',
        'Decision Tree',
        'KNN',
        'Majority Voting',
    ]

    print('10-fold cross-validation:\n')
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(
            estimator=clf,
            X=X_train,
            y=y_train,
            cv=10,
            scoring='roc_auc',
        )
        print(
            "ROC AUC: %0.2f (+/- %0.2f) [%s]" %
            (scores.mean(), scores.std(), label)
        )
    print()

    colors = ['black', 'orange', 'blue', 'green']
    linestyles = [':', '--', '-.', '-']
    for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
        y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(
            fpr,
            tpr,
            color=clr,
            linestyle=ls,
            label="%s (auc = %0.2f)" % (label, roc_auc)
        )

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.show()

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)

    x_min = X_train_std[:, 0].min() - 1
    x_max = X_train_std[:, 0].max() + 1
    y_min = X_train_std[:, 1].min() - 1
    y_max = X_train_std[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1),
    )

    f, axarr = plt.subplots(
        nrows=2,
        ncols=2,
        sharex='col',
        sharey='row',
        figsize=(7, 5),
    )

    for index, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
        clf.fit(X_train_std, y_train)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[index[0], index[1]].contourf(xx, yy, Z, alpha=0.3)
        axarr[index[0], index[1]].scatter(
            X_train_std[y_train == 0, 0],
            X_train_std[y_train == 0, 1],
            c='blue',
            marker='^',
            s=50,
        )
        axarr[index[0], index[1]].scatter(
            X_train_std[y_train == 1, 0],
            X_train_std[y_train == 1, 1],
            c='red',
            marker='o',
            s=50,
        )
        axarr[index[0], index[1]].set_title(tt)

    plt.text(
        -3.5,
        -4.5,
        s='Sepal width [standardized]',
        ha='center',
        va='center',
        fontsize=12,
    )
    plt.text(
        -11.75,
        4.5,
        s='Petal length [standardized]',
        ha='center',
        va='center',
        fontsize=12,
        rotation=90,
    )

    plt.show()

    # print(mv_clf.get_params())

    param_grid = {
        'decisiontreeclassifier__max_depth': [1, 2],
        'pipeline-1__clf__C': [0.001, 0.1, 100.0],
    }

    gs = GridSearchCV(
        estimator=mv_clf,
        param_grid=param_grid,
        cv=10,
        scoring='roc_auc',
    )
    gs.fit(X_train, y_train)

    for params, mean_score, scores in gs.grid_scores_:
        print("%0.3f +/- %0.2f %r" % (mean_score, scores.std() / 2, params))

    print("\nBest parameters: %s" % gs.best_params_)
    print("Accuracy: %.2f" % gs.best_score_)


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, classifiers, vote='classlabel', weights=None):
        if vote not in ('classlabel', 'probability'):
            raise ValueError(
                "vote must be 'classlabel' or 'probability'; got (vote=%r)" %
                vote
            )
        else:
            self.vote = vote
        if (weights is not None) and (len(weights) != len(classifiers)):
            raise ValueError(
                "Number of classifiers and weights must be equal; got %d "
                "weights, %d classifiers" % (len(weights), len(classifiers))
            )
        else:
            self.weights = weights
        self.classifiers = classifiers
        self.named_classifiers = {
            k: v for k, v in _name_estimators(classifiers)
        }

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y)
        self.classes_ = self.label_encoder_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(
                X,
                self.label_encoder_.transform(y)
            )
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        if self.vote == 'classlabel':
            predictions = np.asarray(
                [clf.predict(X) for clf in self.classifiers_]
            ).T
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions,
            )
        elif self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        maj_vote = self.label_encoder_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        probas = np.asarray(
            [clf.predict_proba(X) for clf in self.classifiers_]
        )
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for k, v in six.iteritems(step.get_params(deep=True)):
                    out["%s__%s" % (name, k)] = v
            return out


if __name__ == '__main__':
    # plot_ensemble_error()
    # use_majority_vote_classifier()
    use_bagging_classifier()
