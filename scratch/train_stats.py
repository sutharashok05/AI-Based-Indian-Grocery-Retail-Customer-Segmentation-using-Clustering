import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score

data_path = r"c:\AI-Based Indian Grocery Segmentation using Clustering\data\GroceryDataset_EDA_Final.csv"
models_dir = r"c:\AI-Based Indian Grocery Segmentation using Clustering\models"

df = pd.read_csv(data_path)
features = joblib.load(os.path.join(models_dir, "features.pkl"))
scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
kmeans = joblib.load(os.path.join(models_dir, "kmeans.pkl"))

X = df[features].copy()
X_train, X_test = train_test_split(
    X,
    test_size=0.20,
    random_state=42
)

X_train_scaled = scaler.transform(X_train)
train_clusters = kmeans.predict(X_train_scaled)

train_data = X_train.copy()
train_data["Cluster"] = train_clusters

cluster_profile = train_data.groupby("Cluster").mean()
cluster_counts = train_data["Cluster"].value_counts().sort_index()

sil_score = silhouette_score(X_train_scaled, train_clusters)
db_score = davies_bouldin_score(X_train_scaled, train_clusters)

print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")
print("\nCluster counts:")
print(cluster_counts)
print("\nCluster profiles:")
print(cluster_profile.round(2))

# Also calculate min and max for features in the whole training dataset to guide default input values in frontend!
print("\nTraining set min:")
print(X_train.min())
print("\nTraining set max:")
print(X_train.max())
print("\nTraining set mean:")
print(X_train.mean())
print("\nTraining set median:")
print(X_train.median())
