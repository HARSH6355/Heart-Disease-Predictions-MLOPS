import requests
import time
import random

url = "http://localhost:8080/predict"

base_payload = {
    "age": 54, "sex": 1, "cp": 4, "trestbps": 130, "chol": 250,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3
}

print("Simulating traffic to /predict...")
for i in range(20):
    payload = base_payload.copy()
    # Randomize some features to get a mix of 0 and 1 predictions
    payload["age"] = random.randint(30, 70)
    payload["chol"] = random.randint(150, 350)
    payload["thalach"] = random.randint(100, 200)
    payload["cp"] = random.choice([1, 2, 3, 4])
    
    try:
        response = requests.post(url, json=payload)
        result = response.json()
        print(f"Request {i+1}/20 - Prediction: {result.get('prediction')} - Status: {response.status_code}")
    except Exception as e:
        print(f"Request {i+1} failed: {e}")
    
    time.sleep(0.5)

print("Traffic simulation complete!")
