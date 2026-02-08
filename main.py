import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from kmeans_from_scratch import KMeansFromScratch

# =====================================
# LOAD & STANDARDIZE IRIS DATA
# =====================================
iris = load_iris()
X = iris.data
y_true = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================
# ELBOW METHOD ON IRIS DATA (WCSS)
# =====================================
wcss = []
K_range = range(1, 8)

for k in K_range:
    model = KMeansFromScratch(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    wcss.append(model.wcss_)

plt.figure(figsize=(7, 5))
plt.plot(K_range, wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method on Iris Dataset")
plt.show()

# Optimal K = 3 based on elbow
K_OPT = 3

# =====================================
# CUSTOM K-MEANS (FROM SCRATCH)
# =====================================
custom_kmeans = KMeansFromScratch(n_clusters=K_OPT, random_state=42)
custom_kmeans.fit(X_scaled)

# =====================================
# SILHOUETTE SCORE (QUANTITATIVE METRIC)
# =====================================
custom_silhouette = silhouette_score(X_scaled, custom_kmeans.labels_)
print(f"Silhouette Score (Custom K-Means): {custom_silhouette:.4f}")

# =====================================
# SKLEARN K-MEANS (COMPARISON)
# =====================================
sk_kmeans = KMeans(n_clusters=K_OPT, random_state=42, n_init=10)
sk_labels = sk_kmeans.fit_predict(X_scaled)

sk_silhouette = silhouette_score(X_scaled, sk_labels)
print(f"Silhouette Score (sklearn KMeans): {sk_silhouette:.4f}")

# =====================================
# PCA FOR 2D VISUALIZATION (4D → 2D)
# =====================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(custom_kmeans.centroids)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=custom_kmeans.labels_,
    cmap="viridis",
    s=30
)
plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroids"
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Custom K-Means on Iris (PCA Projection)")
plt.legend()
plt.show()

# =====================================
# CLUSTER DISTRIBUTION
# =====================================
print("\nCluster distribution (Custom K-Means):")
for i in range(K_OPT):
    print(f"Cluster {i}: {np.sum(custom_kmeans.labels_ == i)} samples")