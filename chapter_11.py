from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples


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


def plot_clusters(X, y, km):
    point_info = (
        ('lightgreen', 's', 'cluster 1'),
        ('orange', 'o', 'cluster 2'),
        ('lightblue', 'v', 'cluster 3'),
    )
    cluster_labels = np.unique(y)
    for index, cluster_label in enumerate(cluster_labels):
        c, marker, label = point_info[index]
        plt.scatter(
            X[y == cluster_label, 0],
            X[y == cluster_label, 1],
            s=50,
            c=c,
            marker=marker,
            label=label,
        )
    plt.scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        s=250,
        c='red',
        marker='*',
        label='centroids',
    )

    plt.legend()
    plt.grid()
    plt.show()


def plot_silhouettes(X, y):
    cluster_labels = np.unique(y)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y, metric='euclidean')
    y_ax_lower = 0
    y_ax_upper = 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_clusters)
        plt.barh(
            range(y_ax_lower, y_ax_upper),
            c_silhouette_vals,
            height=1.0,
            edgecolor='none',
            color=color,
        )
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color='red', linestyle='--')

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.show()


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

    km = KMeans(
        n_clusters=3,
        init='random',
        n_init=10,
        max_iter=300,
        tol=1e-04,
        random_state=0,
    )
    y_km = km.fit_predict(X)

    plot_clusters(X, y_km, km)

    print("Distortion: %.2f" % km.inertia_)

    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0,
        )
        km.fit(X)
        distortions.append(km.inertia_)
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')

    plt.show()

    km = KMeans(
        n_clusters=3,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-04,
        random_state=0,
    )
    y_km = km.fit_predict(X)

    plot_silhouettes(X, y_km)

    km = KMeans(
        n_clusters=2,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-04,
        random_state=0,
    )
    y_km = km.fit_predict(X)

    plot_clusters(X, y_km, km)
    plot_silhouettes(X, y_km)


if __name__ == '__main__':
    X_blob, y_blob = get_blob_data()
    plot_blob_data(X_blob)
