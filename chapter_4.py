"""
Copyright Jeremy Nation <jeremy@jeremynation.me>.
Licensed under the MIT license.

Almost entirely copied from code created by Sebastian Raschka, also licensed under the MIT license.

"""
from io import StringIO
from itertools import combinations
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    Imputer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


def work_with_categorical_data():
    df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1'],
    ])
    df.columns = ['color', 'size', 'price', 'class_label']
    print(df, end='\n\n')

    size_mapping = {
        'XL': 3,
        'L': 2,
        'M': 1,
    }
    df['size'] = df['size'].map(size_mapping)
    print(df, end='\n\n')

    inv_size_mapping = {v: k for k, v in size_mapping.items()}
    print(df['size'].map(inv_size_mapping), end='\n\n')

    class_mapping = {
        label: index
        for index, label
        in enumerate(np.unique(df['class_label']))
    }
    print(class_mapping, end='\n\n')
    df['class_label'] = df['class_label'].map(class_mapping)
    print(df, end='\n\n')

    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    df['class_label'] = df['class_label'].map(inv_class_mapping)
    print(df, end='\n\n')

    class_label_encoder = LabelEncoder()
    y = class_label_encoder.fit_transform(df['class_label'].values)
    print(y, end='\n\n')
    class_label_encoder.inverse_transform(y)
    print(class_label_encoder.inverse_transform(y), end='\n\n')

    X = df[['color', 'size', 'price']].values
    color_label_encoder = LabelEncoder()
    X[:, 0] = color_label_encoder.fit_transform(X[:, 0])
    print(X, end='\n\n')

    ohe = OneHotEncoder(categorical_features=[0])
    print(ohe.fit_transform(X).toarray(), end='\n\n')

    print(pd.get_dummies(df[['price', 'color', 'size']]), end='\n\n')


def work_with_numerical_data():
    csv_data = """
        A,B,C,D
        1.0,2.0,3.0,4.0
        5.0,6.0,,8.0
        10.0,11.0,12.0,
    """
    df = pd.read_csv(StringIO(csv_data))
    print(df, end='\n\n')
    print(df.isnull().sum(), end='\n\n')
    print(df.values, end='\n\n')
    print(df.dropna(), end='\n\n')
    print(df.dropna(axis=1), end='\n\n')
    print(df.dropna(how='all'), end='\n\n')
    print(df.dropna(thresh=4), end='\n\n')
    print(df.dropna(subset=['C']), end='\n\n')

    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imr = imr.fit(df)
    imputed_data = imr.transform(df.values)
    print(imputed_data)


def plot_regularization_path(columns, X, y):
    fig = plt.figure()
    ax = plt.subplot(111)

    colors = [
        'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink',
        'lightgreen', 'lightblue', 'gray', 'indigo', 'orange',
    ]

    weights = []
    params = []
    for c in np.arange(-4, 6):
        lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
        lr.fit(X, y)
        weights.append(lr.coef_[1])
        params.append(10**c)

    weights = np.array(weights)

    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(
            params,
            weights[:, column],
            label=columns[column+1],
            color=color,
        )

    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10**-5, 10**5])
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.xscale('log')
    plt.legend(loc='upper left')
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(1.38, 1.03),
        ncol=1,
        fancybox=True,
    )
    plt.show()


def use_sbs_with_knn(columns, X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=2)
    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train, y_train)

    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.show()

    k5 = list(sbs.subsets_[8])
    print(columns[1:][k5])

    knn.fit(X_train, y_train)
    print("Training accuracy: %s" % knn.score(X_train, y_train))
    print("Test accuracy: %s" % knn.score(X_test, y_test))

    knn.fit(X_train[:, k5], y_train)
    print("Training accuracy: %s" % knn.score(X_train[:, k5], y_train))
    print("Test accuracy: %s" % knn.score(X_test[:, k5], y_test))


def plot_feature_importances(columns, X_train, y_train):
    feat_labels = columns[1:]

    forest = RandomForestClassifier(n_estimators=10000, random_state=0)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (
            f+1,
            30,
            feat_labels[indices[f]],
            importances[indices[f]],
        ))
    print()

    plt.title('Feature Importances')
    plt.bar(
        range(X_train.shape[1]),
        importances[indices],
        color='lightblue',
        align='center',
    )
    plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    feature_selector = SelectFromModel(forest, threshold=0.15, prefit=True)
    X_selected = feature_selector.transform(X_train)
    print(X_selected.shape)


def work_with_wine_data():
    df = pd.read_csv(os.path.join('datasets', 'wine.data'), header=None)
    df.columns = [
        'Class label',
        'Alcohol',
        'Malic acid',
        'Ash',
        'Alcalinity of ash',
        'Magnesium',
        'Total phenols',
        'Flavanoids',
        'Nonflavanoid phenols',
        'Proanthocyanins',
        'Color intensity',
        'Hue',
        'OD280/OD315 of diluted wines',
        'Proline',
    ]
    print('Class labels', np.unique(df['Class label']), end='\n\n')
    print(df.head(), end='\n\n')

    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
    )

    ex = pd.DataFrame([0, 1, 2, 3, 4, 5], dtype=np.float64)
    ex[1] = StandardScaler().fit_transform(ex)
    ex[2] = MinMaxScaler().fit_transform(ex[0].reshape(-1, 1))
    ex.columns = ['input', 'standardized', 'normalized']
    print(ex, end='\n\n')

    min_max_scaler = MinMaxScaler()
    X_train_norm = min_max_scaler.fit_transform(X_train)
    X_test_norm = min_max_scaler.transform(X_test)

    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)

    lr = LogisticRegression(penalty='l1', C=0.1)
    lr.fit(X_train_std, y_train)
    print("Training accuracy: %s" % lr.score(X_train_std, y_train))
    print("Test accuracy: %s" % lr.score(X_test_std, y_test))
    print("Intercept: %s" % lr.intercept_)
    print("Coefficients: %s" % lr.coef_)

    # plot_regularization_path(df.columns, X_train_std, y_train)
    # use_sbs_with_knn(df.columns, X_train_std, X_test_std, y_train, y_test)
    plot_feature_importances(df.columns, X_train, y_train)


class SBS(object):

    def __init__(
            self,
            estimator,
            k_features,
            scoring=accuracy_score,
            test_size=0.25,
            random_state=1):
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_, ]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score, ]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]


if __name__ == '__main__':
    # work_with_numerical_data()
    # work_with_categorical_data()
    work_with_wine_data()
