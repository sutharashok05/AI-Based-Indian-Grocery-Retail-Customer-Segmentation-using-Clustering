import pandas as pd
import joblib
import os

data_path = r"c:\AI-Based Indian Grocery Segmentation using Clustering\data\GroceryDataset_EDA_Final.csv"
models_dir = r"c:\AI-Based Indian Grocery Segmentation using Clustering\models"

df = pd.read_csv(data_path)
features = joblib.load(os.path.join(models_dir, "features.pkl"))
scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
kmeans = joblib.load(os.path.join(models_dir, "kmeans.pkl"))

# Map features to column names if they are slightly different in csv
print("CSV columns:", df.columns.tolist())

# Scale and predict
X = df[features]
X_scaled = scaler.transform(X)
df['Cluster'] = kmeans.predict(X_scaled)

print("\nCluster counts:")
print(df['Cluster'].value_counts())

print("\nCluster means:")
print(df.groupby('Cluster')[features].mean())

print("\nOverall means:")
print(df[features].mean())

print("\nMin values by cluster:")
print(df.groupby('Cluster')[features].min())

print("\nMax values by cluster:")
print(df.groupby('Cluster')[features].max())
