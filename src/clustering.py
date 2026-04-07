import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage


def find_optimal_k(X, k_range=range(2, 16)):
    """Compute inertia and silhouette scores across k values on the full feature space.

    k_range capped at 15 for 88 courses to avoid degenerate clusters with
    too few members.
    """
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=100, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))

    best_idx = silhouettes.index(max(silhouettes))
    best_k = list(k_range)[best_idx]
    return list(k_range), inertias, silhouettes, best_k


def run_kmeans(X, k, random_state=42):
    """Run K-Means and return the fitted model."""
    kmeans = KMeans(n_clusters=k, n_init=300, random_state=random_state)
    kmeans.fit(X)
    return kmeans


def get_cluster_assignments(kmeans, labels):
    """Return a dict mapping cluster id to list of course names."""
    clusters = {}
    for i, label in enumerate(labels):
        cluster_id = int(kmeans.labels_[i])
        clusters.setdefault(cluster_id, []).append(label)
    return clusters


def find_similar_courses(input_course, X, labels, n_neighbors=5):
    """Find courses most similar to the input course using nearest neighbors on the full feature space."""
    labels_list = list(labels)
    if input_course not in labels_list:
        return []
    idx = labels_list.index(input_course)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(X)
    distances, indices = nn.kneighbors([X[idx]])
    results = []
    for i, neighbor_idx in enumerate(indices[0][1:]):
        results.append((labels_list[neighbor_idx], distances[0][i + 1]))
    return results


def compute_centroid_distances(kmeans):
    """Compute pairwise Euclidean distances between cluster centroids."""
    centroids = kmeans.cluster_centers_
    dist_vector = pdist(centroids, metric="euclidean")
    dist_matrix = squareform(dist_vector)
    return dist_matrix


def compute_centroid_linkage(kmeans):
    """Hierarchical linkage on cluster centroids for the dendrogram."""
    centroids = kmeans.cluster_centers_
    Z = linkage(centroids, method="ward", metric="euclidean")
    return Z
