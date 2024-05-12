from sklearn.neighbors import NearestNeighbors
import numpy as np

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def _eps_neighborhood(self, point1, point2):
        return np.linalg.norm(point1 - point2) < self.eps

    def _get_neighbors(self, point, X):
        neighbors = []
        for i, neighbor in enumerate(X):
            if self._eps_neighborhood(point, neighbor):
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, labels, point, neighbors, cluster_id):
        labels[point] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
            if labels[neighbor] == 0:
                labels[neighbor] = cluster_id
                new_neighbors = self._get_neighbors(X[neighbor], X)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors
            i = i + 1

    def fit_predict(self, X):
        n_samples = len(X)
        labels = np.zeros(n_samples, dtype=int)
        cluster_id = 0

        for i in range(n_samples):
            if labels[i] != 0:
                continue
            neighbors = self._get_neighbors(X[i], X)
            if len(neighbors) < self.min_samples:
                labels[i] = -1
            else:
                cluster_id += 1
                self._expand_cluster(X, labels, i, neighbors, cluster_id)

        return labels
