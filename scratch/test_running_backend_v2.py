import requests

base_url = "http://127.0.0.1:8000"

print("--- Testing GET / ---")
r_home = requests.get(f"{base_url}/")
print("Status:", r_home.status_code)
print("Response:", r_home.json())
assert r_home.status_code == 200

print("\n--- Testing GET /health ---")
r_health = requests.get(f"{base_url}/health")
print("Status:", r_health.status_code)
print("Response:", r_health.json())
assert r_health.status_code == 200

print("\n--- Testing GET /model-info ---")
r_info = requests.get(f"{base_url}/model-info")
print("Status:", r_info.status_code)
info = r_info.json()
print("Keys:", list(info.keys()))
assert r_info.status_code == 200
assert "curves" in info
assert "overall_stats" in info
assert "clusters" in info

print("\n--- Testing GET /analytics-data ---")
r_analytics = requests.get(f"{base_url}/analytics-data")
print("Status:", r_analytics.status_code)
analytics = r_analytics.json()
print("Count of items:", analytics["count"])
print("First item keys:", list(analytics["data"][0].keys()))
assert r_analytics.status_code == 200
assert "PCA1" in analytics["data"][0]
assert "PCA2" in analytics["data"][0]

print("\n--- Testing POST /predict ---")
payload = {
    "price": 185.00,
    "discount": 3.5,
    "rating": 4.8,
    "reviews": 850,
    "title_length": 75,
    "feature_length": 190,
    "description_length": 1850
}
r_predict = requests.post(f"{base_url}/predict", json=payload)
print("Status:", r_predict.status_code)
pred = r_predict.json()
print("Response keys:", list(pred.keys()))
print("Predicted Cluster:", pred["prediction"]["cluster_id"])
print("Cluster Name:", pred["prediction"]["cluster_name"])
print("Proximity:", pred["prediction"]["proximity"])
print("Proximity Details:", pred["cluster_proximity"])
print("Insights:", pred["insights"])
assert r_predict.status_code == 200
assert pred["success"] is True
assert "prediction" in pred
assert "cluster_profile" in pred
assert "cluster_proximity" in pred
assert "insights" in pred

print("\nAll integration V2 checks passed successfully!")
