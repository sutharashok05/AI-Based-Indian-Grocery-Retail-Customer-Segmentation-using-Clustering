import sys
import os

# Add backend directory to path
sys.path.append(r"c:\AI-Based Indian Grocery Segmentation using Clustering")

from backend.main import predict_cluster, model_info, health, home
from backend.schemas import ProductInput

# Run home, health, model_info
print("Home:", home())
print("Health:", health())
print("Model Info (keys):", model_info().keys())

# Test prediction with a budget product
sample_budget = ProductInput(
    price=10.0,
    discount=0.0,
    rating=4.2,
    reviews=50,
    title_length=40,
    feature_length=30,
    description_length=150
)
res_budget = predict_cluster(sample_budget)
print("\nBudget Product Prediction:")
print("Predicted Cluster:", res_budget['cluster'])
print("Cluster Name:", res_budget['cluster_name'])
print("Distances:", res_budget['distances'])
print("Proximities:", res_budget['proximity_percentages'])

# Test prediction with a premium product
sample_premium = ProductInput(
    price=250.0,
    discount=5.0,
    rating=4.8,
    reviews=600,
    title_length=80,
    feature_length=200,
    description_length=2000
)
res_premium = predict_cluster(sample_premium)
print("\nPremium Product Prediction:")
print("Predicted Cluster:", res_premium['cluster'])
print("Cluster Name:", res_premium['cluster_name'])
print("Distances:", res_premium['distances'])
print("Proximities:", res_premium['proximity_percentages'])
print("Success!")
