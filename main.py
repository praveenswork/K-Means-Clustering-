import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler

from kmeans_from_scratch import KMeansFromScratch

# =====================================
# TASK 2: SYNTHETIC DATA (INITIAL TEST)
# =====================================
X_syn, _ = make_blobs(
    n_samples=400,
    centers=4,
    n_features=2,
    cluster_std=1.0,
    random_state=42
)

# =====================================
# TASK 3: ELBOW METHOD (WCSS, MANUAL)
# =====================================
wcss_values = []
K_range = range(1, 8)

for k in K_range:
    kmeans = KMeansFromScratch(n_clusters=k, random_state=42)
    kmeans.fit(X_syn)
    wcss_values.append(kmeans.wcss_)

plt.figure(figsize=(7, 5))
plt.plot(K_range, wcss_values, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method (WCSS vs K)")
plt.show()

# From elbow plot, K = 3 is chosen for Iris

# =====================================
# TASK 2 & 4: IRIS DATASET
# =====================================
iris = load_iris()
X_iris = iris.data
y_true = iris.target
feature_names = iris.feature_names

# Standardize features
X_iris = StandardScaler().fit_transform(X_iris)

# Apply Custom K-Means on Iris
kmeans_iris = KMeansFromScratch(n_clusters=3, random_state=42)
kmeans_iris.fit(X_iris)

# =====================================
# TASK 4: 2D VISUALIZATION
# =====================================
# Using Sepal Length (0) vs Sepal Width (1)
plt.figure(figsize=(8, 6))
plt.scatter(
    X_iris[:, 0],
    X_iris[:, 1],
    c=kmeans_iris.labels_,
    cmap="viridis",
    s=30
)
plt.scatter(
    kmeans_iris.centroids[:, 0],
    kmeans_iris.centroids[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroids"
)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("Custom K-Means on Iris Dataset (2D Projection)")
plt.legend()
plt.show()

# =====================================
# TASK 5: TEXTUAL ANALYSIS SUPPORT
# =====================================
print("Cluster label distribution:")
for i in range(3):
    print(f"Cluster {i}: {np.sum(kmeans_iris.labels_ == i)} samples")

print("\nTrue Iris species distribution:")
for i, name in enumerate(iris.target_names):
    print(f"{name}: {np.sum(y_true == i)} samples")