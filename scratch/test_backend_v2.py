import sys
import os

# Add workspace to python path
sys.path.append(r"c:\AI-Based Indian Grocery Segmentation using Clustering")

from backend.main import predict_cluster, model_info, health, home, analytics_data
from backend.schemas import ProductInput

# Run base methods
print("Home:", home())
print("Health:", health())
info = model_info()
print("Model Info Keys:", info.keys())
print("Clusters C0 name:", info["clusters"]["0"]["name"])
print("Curves keys:", info["curves"].keys())

analytics = analytics_data()
print("Analytics Data Count:", analytics["count"])
print("First Data Record Keys:", list(analytics["data"][0].keys()))
print("First Data Record PCA:", analytics["data"][0]["PCA1"], analytics["data"][0]["PCA2"])

# Run prediction
product = ProductInput(
    price=120.0,
    discount=5.0,
    rating=4.6,
    reviews=400,
    title_length=65,
    feature_length=150,
    description_length=1200
)

pred_res = predict_cluster(product)
print("\nPrediction Result:")
import json
print(json.dumps(pred_res, indent=2))
assert pred_res["success"] is True
assert "prediction" in pred_res
assert "cluster_profile" in pred_res
assert "cluster_proximity" in pred_res
assert "insights" in pred_res
print("\nAll backend V2 checks passed successfully!")
