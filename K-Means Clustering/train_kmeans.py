"""
Train a K-Means clustering model on `Mall_Customers.csv` and
serialize the model plus training data to `kmeans_model.pkl`.

Saved pickle format (dict):
- 'model': trained KMeans object
- 'X_train': numpy array of training features
- 'y_train': cluster labels assigned by the model

This file is commented for clarity and intended to be run from the
`K-Means Clustering/Python` directory or the K-Means root where the CSV sits.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# Load dataset (expects Mall_Customers.csv in same directory)
csv_path = os.path.join(os.path.dirname(__file__), 'Python', 'Mall_Customers.csv')
if not os.path.exists(csv_path):
    # fallback to repository root CSV location
    csv_path = os.path.join(os.path.dirname(__file__), 'Python', 'Mall_Customers.csv')

print(f"Loading data from: {csv_path}")
df = pd.read_csv(csv_path)

# Select features: Annual Income (k$) and Spending Score (1-100)
X = df.iloc[:, [3, 4]].values

# Choose number of clusters; this can be tuned after inspecting dendrogram/Elbow plot
n_clusters = 5

# Train KMeans model
print(f"Training KMeans with n_clusters={n_clusters}...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_km = kmeans.fit_predict(X)

# Optional: save cluster visualization
plt.figure(figsize=(8, 5))
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'yellow']
for i in range(n_clusters):
    plt.scatter(X[y_km == i, 0], X[y_km == i, 1], s=50, c=colors[i % len(colors)], label=f'Cluster {i+1}')
# plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.title('K-Means Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.tight_layout()
plt.savefig('kmeans_clusters.png')
plt.close()
print("Saved cluster visualization to 'kmeans_clusters.png'")

# Serialize model and training data together so the Flask app can compute cluster centers
model_data = {
    'model': kmeans,
    'X_train': X,
    'y_train': y_km
}

out_path = os.path.join(os.path.dirname(__file__), 'Python', 'kmeans_model.pkl')
# Save next to the dataset so Flask code can find it; also write a copy here
with open(out_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"Saved kmeans model and training data to: {out_path}")

# also save a copy in the flask app directory for convenience
flask_copy = os.path.join(os.path.dirname(__file__), 'flask_kmeans_app', 'kmeans_model.pkl')
try:
    with open(flask_copy, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Copied model to flask app directory: {flask_copy}")
except Exception:
    print("Could not copy model into flask app directory automatically; please copy manually if needed.")
