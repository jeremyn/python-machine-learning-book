from io import StringIO

import pandas as pd
from sklearn.preprocessing import Imputer


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


if __name__ == '__main__':
    work_with_numerical_data()
