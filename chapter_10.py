import os

import pandas as pd


def get_housing_data():
    df = pd.read_csv(
        os.path.join('datasets', 'housing.data'),
        header=None,
        sep='\s+',
    )
    df.columns = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
        'PTRATIO', 'B', 'LSTAT', 'MEDV',
    ]
    return df


if __name__ == '__main__':
    df = get_housing_data()
    print(df.head())
