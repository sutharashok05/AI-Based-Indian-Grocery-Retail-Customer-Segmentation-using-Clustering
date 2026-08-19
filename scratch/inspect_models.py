import joblib
import os

models_dir = r"c:\AI-Based Indian Grocery Segmentation using Clustering\models"
features = joblib.load(os.path.join(models_dir, "features.pkl"))
scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
kmeans = joblib.load(os.path.join(models_dir, "kmeans.pkl"))

print("Features:", features)
print("Type of features:", type(features))
print("Scaler mean_:", getattr(scaler, "mean_", None))
print("Scaler scale_:", getattr(scaler, "scale_", None))
print("KMeans cluster centers:", kmeans.cluster_centers_)
print("KMeans labels shape/count:", getattr(kmeans, "labels_", None).shape if getattr(kmeans, "labels_", None) is not None else "None")
print("KMeans n_clusters:", kmeans.n_clusters)
