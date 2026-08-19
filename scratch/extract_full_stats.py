import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split

data_path = r"c:\AI-Based Indian Grocery Segmentation using Clustering\data\GroceryDataset_EDA_Final.csv"
models_dir = r"c:\AI-Based Indian Grocery Segmentation using Clustering\models"

df = pd.read_csv(data_path)
features = joblib.load(os.path.join(models_dir, "features.pkl"))
scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
kmeans = joblib.load(os.path.join(models_dir, "kmeans.pkl"))

# Scale and predict on the entire dataset to get overall and cluster stats
X = df[features]
X_scaled = scaler.transform(X)
df['Cluster'] = kmeans.predict(X_scaled)

print("=== OVERALL DATASET STATS ===")
print("Total Products:", len(df))
print("Avg Price:", df['Price_num'].mean())
print("Avg Rating:", df['Rating_num'].mean())
print("Avg Discount:", df['Discount_num'].mean())
print("Avg Reviews:", df['Reviews_num'].mean())

print("\n=== CLUSTER DISTRIBUTION ===")
counts = df['Cluster'].value_counts()
for cluster_id in sorted(counts.index):
    count = counts[cluster_id]
    pct = (count / len(df)) * 100
    print(f"Cluster {cluster_id}: {count} products ({pct:.2f}%)")

print("\n=== FEATURE MEANS BY CLUSTER ===")
profile = df.groupby('Cluster')[features].mean()
print(profile.round(4))

print("\n=== TOP CATEGORIES BY CLUSTER ===")
for cluster_id in sorted(df['Cluster'].unique()):
    print(f"\nCluster {cluster_id} Top 5 Categories:")
    top_cats = df[df['Cluster'] == cluster_id]['Sub Category'].value_counts().head(5)
    for cat, count in top_cats.items():
        pct = (count / len(df[df['Cluster'] == cluster_id])) * 100
        print(f" - {cat}: {count} ({pct:.2f}%)")
