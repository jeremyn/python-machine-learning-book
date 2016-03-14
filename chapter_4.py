from io import StringIO

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    Imputer,
    LabelEncoder,
    OneHotEncoder,
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


if __name__ == '__main__':
    #work_with_numerical_data()
    work_with_categorical_data()
