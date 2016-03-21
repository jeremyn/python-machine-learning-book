import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


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
    X_rm = df[['RM']].values
    y = df['MEDV'].values
    return df, X_rm, y


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


class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        y = y.ravel()
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='lightblue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)


def plot_custom_linear_model(X, y):
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_X.fit_transform(X)
    y_std = sc_y.fit_transform(y.reshape(-1, 1))

    lr = LinearRegressionGD()
    lr.fit(X_std, y_std)

    plt.plot(range(1, lr.n_iter+1), lr.cost_)
    plt.ylabel('SSE')
    plt.xlabel('Epoch')

    plt.show()

    lin_regplot(X_std, y_std, lr)
    plt.xlabel('Average number of rooms [RM] (standardized)')
    plt.ylabel("Price in $1000's [MEDV] (standardized)")

    plt.show()

    print("Slope: %.3f" % lr.w_[1])
    print("Intercept: %.3f" % lr.w_[0])

    num_rooms_std = sc_X.transform([[5.0]])
    price_std = lr.predict(num_rooms_std)
    print("Price in $1000's: %.3f" % sc_y.inverse_transform(price_std))


def plot_sklearn_linear_model(X, y):
    sklearn_lr = LinearRegression()
    sklearn_lr.fit(X, y)
    print("Slope: %.3f" % sklearn_lr.coef_[0])
    print("Intercept: %.3f" % sklearn_lr.intercept_)

    lin_regplot(X, y, sklearn_lr)
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel("Price in $1000's [MEDV]")

    plt.show()


def create_linear_model_with_normal_equation(X, y):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    w = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
    print("Slope: %.3f" % w[1])
    print("Intercept: %.3f" % w[0])


if __name__ == '__main__':
    df, X_rm, y = get_housing_data()
    # visualize_housing_data(df)
    # plot_custom_linear_model(X_rm, y)
    # plot_sklearn_linear_model(X_rm, y)
    create_linear_model_with_normal_equation(X_rm, y)
