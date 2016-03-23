from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import (
    dendrogram,
    linkage,
)
from scipy.spatial.distance import (
    pdist,
    squareform,
)
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
)
from sklearn.datasets import (
    make_blobs,
    make_moons,
)
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


def process_random_data():
    variables = ['X', 'Y', 'Z']
    labels = ["ID_%s" % i for i in range(5)]
    X = np.random.random_sample([5, 3])*10
    df = pd.DataFrame(X, columns=variables, index=labels)
    print(df, end='\n\n')

    row_dist = pd.DataFrame(
        squareform(pdist(df, metric='euclidean')),
        columns=labels,
        index=labels,
    )

    row_clusters_options = (
        linkage(row_dist, method='complete', metric='euclidean'),  # bad
        linkage(pdist(df, metric='euclidean'), method='complete'),  # good
        linkage(df.values, method='complete', metric='euclidean'),  # good
    )

    columns = (
        'row label 1',
        'row label 2',
        'distance',
        'no. of items in clust.',
    )
    for row_clusters_option in row_clusters_options:
        index = [
            "cluster %d" % (i+1) for i in range(row_clusters_option.shape[0])
        ]
        print(
            pd.DataFrame(
                row_clusters_option,
                columns=columns,
                index=index,
            ),
            end='\n\n',
        )

    row_clusters = row_clusters_options[2]

    dendrogram(row_clusters, labels=labels)
    plt.ylabel('Euclidean distance')

    plt.show()

    fig = plt.figure(figsize=(8, 8), facecolor='white')
    axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    row_dendr = dendrogram(row_clusters, orientation='left')

    df_rowclust = df.ix[row_dendr['leaves'][::-1]]

    axd.set_xticks([])
    axd.set_yticks([])
    for i in axd.spines.values():
        i.set_visible(False)

    axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
    cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)
    axm.set_xticklabels([''] + list(df_rowclust.columns))
    axm.set_yticklabels([''] + list(df_rowclust.index))

    plt.show()

    ac = AgglomerativeClustering(
        n_clusters=2,
        affinity='euclidean',
        linkage='complete',
    )
    labels = ac.fit_predict(X)
    print("Cluster labels: %s" % labels)


def process_moon_data():
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    km = KMeans(n_clusters=2, random_state=0)
    y_km = km.fit_predict(X)
    ax1.scatter(
        X[y_km == 0, 0],
        X[y_km == 0, 1],
        c='lightblue',
        marker='o',
        s=40,
        label='cluster 1',
    )
    ax1.scatter(
        X[y_km == 1, 0],
        X[y_km == 1, 1],
        c='red',
        marker='s',
        s=40,
        label='cluster 2',
    )
    ax1.set_title('K-means clustering')

    ac = AgglomerativeClustering(
        n_clusters=2,
        affinity='euclidean',
        linkage='complete',
    )
    y_ac = ac.fit_predict(X)
    ax2.scatter(
        X[y_ac == 0, 0],
        X[y_ac == 0, 1],
        c='lightblue',
        marker='o',
        s=40,
        label='cluster 1',
    )
    ax2.scatter(
        X[y_ac == 1, 0],
        X[y_ac == 1, 1],
        c='red',
        marker='s',
        s=40,
        label='cluster 2',
    )
    ax2.set_title('Agglomerative clustering')

    plt.legend()
    plt.show()

    db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
    y_db = db.fit_predict(X)
    plt.scatter(
        X[y_db == 0, 0],
        X[y_db == 0, 1],
        c='lightblue',
        marker='o',
        s=40,
        label='cluster 1',
    )
    plt.scatter(
        X[y_db == 1, 0],
        X[y_db == 1, 1],
        c='red',
        marker='s',
        s=40,
        label='cluster 2',
    )

    plt.legend()
    plt.show()


if __name__ == '__main__':
    X_blob, y_blob = get_blob_data()
    # plot_blob_data(X_blob)
    np.random.seed(123)
    # process_random_data()
    process_moon_data()
