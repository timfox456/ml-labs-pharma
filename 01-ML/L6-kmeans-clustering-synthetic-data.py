# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# # ---

# # Lab: K-Means Clustering for Unsupervised Learning
#
# **Goal:** In this lab, you will explore unsupervised learning by using the K-Means algorithm to find clusters in a dataset where you don't have any labels.
#
# **Key Concepts:**
# - **Unsupervised Learning:** Finding patterns in data without pre-existing labels.
# - **Clustering:** Grouping similar data points together.
# - **K-Means Algorithm:** An iterative algorithm for partitioning data into K clusters.
#
# ---
#
# ## 1. Setup
#
# We need `scikit-learn` to generate data and for the K-Means model, and `matplotlib` to visualize the results.

# +
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# -

# ## 2. Generate Synthetic Data
#
# For clustering, it's often easiest to learn with synthetic data because we know the "ground truth." We can generate data with a specific number of clusters and then see if the K-Means algorithm can find them.
#
# We will generate 300 data points grouped into 4 distinct clusters.

# +
# Generate the data
# X contains the coordinates, y_true contains the true cluster labels (which we'll ignore for training)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.70, random_state=0)

# Plot the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Generated Data (True Labels Hidden)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
# -

# **The Challenge:** The plot above is what the unsupervised algorithm sees. It has no colors or labels. Its job is to figure out the underlying groups on its own.
#
# ---
#
# ## 3. Build and Train the K-Means Model
#
# We will initialize the `KMeans` model, telling it that we expect to find 4 clusters (`n_clusters=4`). Then, we'll fit it to our data `X`.

# +
# Initialize the K-Means model
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10) # n_init='auto' is the default in newer versions

# Train the model (fit it to the data)
print("Fitting the K-Means model...")
kmeans.fit(X)
print("Model fitting complete.")
# -

# ## 4. Analyze the Results
#
# The trained model has now assigned a cluster label to each data point and has calculated the center of each cluster.
#
# ### Get the Predictions and Cluster Centers

# +
# Get the cluster assignments for each data point
y_kmeans = kmeans.predict(X)

# Get the coordinates of the cluster centers
centers = kmeans.cluster_centers_
# -

# ### Visualize the Discovered Clusters
# Now, let's plot the data again, but this time we'll color the points based on the cluster labels found by the K-Means algorithm. We'll also plot the cluster centers.

# +
plt.figure(figsize=(10, 8))
# Plot the data points, colored by their assigned cluster
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plot the cluster centers as large red circles
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Cluster Centers')

plt.title("Clusters Discovered by K-Means")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
# -

# **Analysis:** As you can see, the K-Means algorithm did an excellent job of identifying the four original clusters we created. It successfully partitioned the data into meaningful groups without being given any labels beforehand.
#
# ---
#
# ## Conclusion
#
# In this lab, you have performed your first unsupervised learning task. You learned:
# 1.  The goal of clustering is to find inherent groups in unlabeled data.
# 2.  How to use `make_blobs` to create synthetic data for testing clustering algorithms.
# 3.  How to train a `KMeans` model to partition the data.
# 4.  How to visualize the results to verify that the algorithm has found the underlying structure.
