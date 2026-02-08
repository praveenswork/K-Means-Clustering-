import numpy as np

class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            points = X[labels == k]
            if len(points) > 0:
                centroids[k] = points.mean(axis=0)
        return centroids

    def _compute_wcss(self, X, labels, centroids):
        wcss = 0.0
        for k in range(self.n_clusters):
            points = X[labels == k]
            wcss += np.sum((points - centroids[k]) ** 2)
        return wcss

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X, self.centroids)
            new_centroids = self._update_centroids(X, labels)

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

        self.labels_ = labels
        self.wcss_ = self._compute_wcss(X, labels, self.centroids)
        return self