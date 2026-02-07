import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from kmeans_from_scratch import KMeansFromScratch

# Generate 5D synthetic dataset
X, _ = make_blobs(
    n_samples=600,
    centers=4,
    n_features=5,   # 🔴 IMPORTANT FIX
    cluster_std=1.2,
    random_state=42
)

# -------------------------------
# Elbow Method (Task Requirement)
# -------------------------------
inertias = []
K_range = range(1, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(K_range, inertias, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# -------------------------------
# Custom K-Means (K=4)
# -------------------------------
custom_kmeans = KMeansFromScratch(n_clusters=4, random_state=42)
custom_kmeans.fit(X)

# Visualization (only first 2 dimensions)
plt.figure(figsize=(8, 6))
plt.scatter(
    X[:, 0], X[:, 1],
    c=custom_kmeans.labels_,
    cmap="viridis",
    s=30
)
plt.title("Custom K-Means Clustering (First 2 Dimensions)")
plt.show()

# -------------------------------
# Task 4: Silhouette Score
# -------------------------------
custom_score = silhouette_score(X, custom_kmeans.labels_)

sk_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
sk_labels = sk_kmeans.fit_predict(X)
sk_score = silhouette_score(X, sk_labels)

print("Silhouette Scores")
print("-----------------")
print(f"Custom K-Means  : {custom_score:.4f}")
print(f"Sklearn K-Means : {sk_score:.4f}")