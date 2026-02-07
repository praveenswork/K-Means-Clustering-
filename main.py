import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from kmeans_from_scratch import KMeansFromScratch

# Generate synthetic dataset
X, _ = make_blobs(
    n_samples=600,
    centers=4,
    cluster_std=1.0,
    random_state=42
)

# Custom K-Means
custom_kmeans = KMeansFromScratch(n_clusters=4, random_state=42)
custom_kmeans.fit(X)

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=custom_kmeans.labels_, cmap="viridis", s=30)
plt.scatter(
    custom_kmeans.centroids[:, 0],
    custom_kmeans.centroids[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroids"
)
plt.title("Custom K-Means Clustering (K=4)")
plt.legend()
plt.show()

# Task 4: Silhouette Score comparison
custom_score = silhouette_score(X, custom_kmeans.labels_)

sk_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
sk_labels = sk_kmeans.fit_predict(X)
sk_score = silhouette_score(X, sk_labels)

print("Silhouette Scores")
print("-----------------")
print(f"Custom K-Means  : {custom_score:.4f}")
print(f"Sklearn K-Means : {sk_score:.4f}")