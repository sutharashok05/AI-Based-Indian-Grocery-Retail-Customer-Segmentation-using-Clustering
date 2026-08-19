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
print("Keys:", r_info.json().keys())
assert r_info.status_code == 200

print("\n--- Testing GET /analytics-data ---")
r_analytics = requests.get(f"{base_url}/analytics-data")
print("Status:", r_analytics.status_code)
print("Count of items:", r_analytics.json()["count"])
assert r_analytics.status_code == 200

print("\n--- Testing POST /predict ---")
payload = {
    "price": 12.5,
    "discount": 0.5,
    "rating": 4.3,
    "reviews": 120,
    "title_length": 45,
    "feature_length": 80,
    "description_length": 250
}
r_predict = requests.post(f"{base_url}/predict", json=payload)
print("Status:", r_predict.status_code)
print("Response keys:", r_predict.json().keys())
print("Predicted Cluster:", r_predict.json()["cluster"])
print("Cluster Name:", r_predict.json()["cluster_name"])
print("Distances:", r_predict.json()["distances"])
print("Proximities:", r_predict.json()["proximity_percentages"])
assert r_predict.status_code == 200

print("\nAll integration checks passed successfully!")
