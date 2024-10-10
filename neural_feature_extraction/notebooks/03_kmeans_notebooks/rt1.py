# %%

import sys
sys.path.append('../')


from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from helper import extract_embeddings

# %%
sbert_explained_variance_95 = {
    "alfred": 165,
    "scout": 194,
    "rt1": 27,
    "bridge": 115,
    "tacoplay": 31
}

# %%
model_name = "all-mpnet-base-v2" # sbert
dataset_name = "rt1"

results_fp = f"/home/slwanna/neural-feature-extraction/neural_feature_extraction/notebooks/results/01_{model_name}_{dataset_name}"

embeddings = extract_embeddings(f"{results_fp}.csv")

data = np.array(embeddings)

data.shape

# %%
sse = []
k_start = 2
k_end = 70
clusters_range = range(k_start, k_end)

for k in clusters_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)

# visualize results
plt.plot(clusters_range, sse,  label='No PCA Reduction')
plt.legend()
plt.xticks(np.arange(k_start, k_end, step=10))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")

# %%
data = np.array(embeddings)
pca=PCA(n_components=sbert_explained_variance_95[dataset_name])
dim_reduced_embeddings = pca.fit_transform(data)

sse = []
X_dim_red = normalize(dim_reduced_embeddings, norm='l2')
k_start = 2
k_end = 70
clusters_range = range(k_start, k_end)

for k in clusters_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_dim_red)
    sse.append(kmeans.inertia_)

# visualize results
plt.plot(clusters_range, sse, label='PCA Reduction')
plt.legend()

plt.xticks(np.arange(k_start, k_end, step=10))
plt.xlabel("Dim Reduction Number of Clusters")
plt.ylabel("SSE")
plt.show()

# %%
# Fit KMeans with n clusters
N_CLUSTERS = 6

kmeans = KMeans(n_clusters=N_CLUSTERS)
kmeans.fit(embeddings)

# Get cluster centers and labels for each point
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the clusters
ax.scatter(X_dim_red[:, 0], X_dim_red[:, 1], X_dim_red[:, 2], c=labels, s=10, cmap='viridis')

# Plot the centroids
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', s=75, alpha=0.75, marker='x')

# Set plot title and labels
ax.set_title('KMeans Clustering - No Dim Reduction')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

# Show the plot
plt.show()

# %%
original_df = pd.read_csv(f"{results_fp}.csv")
original_df["kmeans_labels"] = labels

# %%
original_df.info()

# %%
original_df.groupby('kmeans_labels')['nl_command_exs'].value_counts(normalize=True).groupby(level=0).head(5)

# %%
original_df.groupby('kmeans_labels')['nl_command_exs'].value_counts().groupby(level=0).head(5)

# %%
# Step 1: Store the results in a variable
result = original_df.groupby('kmeans_labels')['nl_command_exs'].value_counts().groupby(level=0).head(15)

# Step 2: Write the result to a .txt file
with open(f"/home/slwanna/neural-feature-extraction/neural_feature_extraction/notebooks/results/{model_name}_{dataset_name}_kmeans_membership_output.txt", "w") as file:
    file.write(result.to_string())


