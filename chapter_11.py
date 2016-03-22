import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def get_blob_data():
    X, y = make_blobs(
        n_samples=150,
        n_features=2,
        centers=3,
        cluster_std=0.5,
        shuffle=True,
        random_state=0,
    )
    return X, y


def plot_blob_data(X):
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c='white',
        marker='o',
        s=50,
    )
    plt.grid()

    plt.show()


if __name__ == '__main__':
    X_blob, y_blob = get_blob_data()
    plot_blob_data(X_blob)
