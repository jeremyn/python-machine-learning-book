import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def visualize_housing_data(df):
    sns.set(style='whitegrid', context='notebook')
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

    sns.pairplot(df[cols], size=2.5)

    plt.show()

    correlation_matrix = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    heatmap = sns.heatmap(
        correlation_matrix,
        cbar=True,
        annot=True,
        square=True,
        fmt='.2f',
        annot_kws={'size': 15},
        yticklabels=cols,
        xticklabels=cols,
    )

    plt.show()

    sns.reset_orig()


if __name__ == '__main__':
    df = get_housing_data()
    visualize_housing_data(df)
