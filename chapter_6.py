import os

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder


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
