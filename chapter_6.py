import os

import numpy as np
import pandas as pd
from sklearn.cross_validation import (
    StratifiedKFold,
    train_test_split,
)
from sklearn.decomposition import PCA
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


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_wdbc_data()

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
    print("\nCV accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))
