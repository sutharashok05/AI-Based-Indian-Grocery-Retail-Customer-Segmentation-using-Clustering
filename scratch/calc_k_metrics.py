import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data_path = r"c:\AI-Based Indian Grocery Segmentation using Clustering\data\GroceryDataset_EDA_Final.csv"
models_dir = r"c:\AI-Based Indian Grocery Segmentation using Clustering\models"

df = pd.read_csv(data_path)
features = joblib.load(os.path.join(models_dir, "features.pkl"))
scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

X = df[features].copy()
X_train, X_test = train_test_split(
    X,
    test_size=0.20,
    random_state=42
)

X_train_scaled = scaler.transform(X_train)

inertia_vals = {}
sil_vals = {}

for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_train_scaled)
    inertia_vals[k] = float(km.inertia_)
    sil_vals[k] = float(silhouette_score(X_train_scaled, km.labels_))

print("Inertia:", inertia_vals)
print("Silhouette:", sil_vals)
