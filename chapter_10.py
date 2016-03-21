import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    RANSACRegressor,
)
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
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
    X = df.iloc[:, :-1].values
    y = df['MEDV'].values
    return df, X, y


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


def plot_ransac_linear_model(X, y):
    ransac = RANSACRegressor(
        LinearRegression(),
        max_trials=100,
        min_samples=50,
        residual_metric=lambda x: np.sum(np.abs(x), axis=1),
        residual_threshold=5.0,
        random_state=0,
    )
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_X = np.arange(3, 10, 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])

    plt.scatter(
        X[inlier_mask],
        y[inlier_mask],
        c='blue',
        marker='o',
        label='Inliers',
    )
    plt.scatter(
        X[outlier_mask],
        y[outlier_mask],
        c='lightgreen',
        marker='s',
        label='Outliers',
    )
    plt.plot(line_X, line_y_ransac, color='red')
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel("Price in $1000's [MEDV]")
    plt.legend(loc='upper left')

    plt.show()

    print("Slope: %.3f" % ransac.estimator_.coef_[0])
    print("Intercept: %.3f" % ransac.estimator_.intercept_)


def evaluate_sklearn_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
    )
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    plt.scatter(
        y_train_pred,
        y_train_pred - y_train,
        c='blue',
        marker='o',
        label='Training data',
    )
    plt.scatter(
        y_test_pred,
        y_test_pred - y_test,
        c='lightgreen',
        marker='s',
        label='Test data',
    )
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
    plt.xlim([-10, 50])

    plt.show()

    print("MSE train: %.3f, test: %.3f" % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred),
    ))
    print("R^2 train: %.3f, test: %.3f" % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred),
    ))


if __name__ == '__main__':
    df, X, y = get_housing_data()
    X_rm = df[['RM']].values
    # visualize_housing_data(df)
    sns.reset_orig()
    # plot_custom_linear_model(X_rm, y)
    # plot_sklearn_linear_model(X_rm, y)
    # create_linear_model_with_normal_equation(X_rm, y)
    # plot_ransac_linear_model(X_rm, y)
    evaluate_sklearn_linear_regression(X, y)
