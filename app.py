import threading
import time
import webbrowser

from flask import Flask, jsonify, render_template, request
from prometheus_client import Counter, Histogram, generate_latest

from src.logger import logging
from src.pipeline.predict_pipeline import HeartDiseaseInput, PredictPipeline


app = Flask(__name__)

REQUEST_COUNT = Counter(
    "heart_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "heart_api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
)
PREDICTION_COUNT = Counter(
    "heart_api_predictions_total",
    "Total predictions by class",
    ["prediction"],
)


@app.after_request
def record_request(response):
    REQUEST_COUNT.labels(request.path, request.method, response.status_code).inc()
    return response


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain; version=0.0.4"}


@app.route("/predict", methods=["POST"])
def predict():
    started_at = time.perf_counter()

    try:
        payload = request.get_json(silent=True) or request.form.to_dict()
        input_data = HeartDiseaseInput(**_coerce_payload(payload))
        result = PredictPipeline().predict(input_data.to_dataframe())

        PREDICTION_COUNT.labels(str(result["prediction"])).inc()
        logging.info("Prediction request completed: %s", result)

        return jsonify(
            {
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "risk_label": "heart disease risk" if result["prediction"] == 1 else "no heart disease risk",
            }
        )
    except Exception as error:
        logging.exception("Prediction request failed")
        return jsonify({"error": str(error)}), 400
    finally:
        REQUEST_LATENCY.labels("/predict").observe(time.perf_counter() - started_at)


def _coerce_payload(payload: dict) -> dict:
    required_fields = HeartDiseaseInput.__dataclass_fields__.keys()
    missing_fields = [field for field in required_fields if field not in payload]

    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    return {
        "age": float(payload["age"]),
        "sex": int(payload["sex"]),
        "cp": int(payload["cp"]),
        "trestbps": float(payload["trestbps"]),
        "chol": float(payload["chol"]),
        "fbs": int(payload["fbs"]),
        "restecg": int(payload["restecg"]),
        "thalach": float(payload["thalach"]),
        "exang": int(payload["exang"]),
        "oldpeak": float(payload["oldpeak"]),
        "slope": int(payload["slope"]),
        "ca": float(payload["ca"]),
        "thal": float(payload["thal"]),
    }


def _open_browser():
    """Open the app in the default browser after a short delay."""
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    # Open browser automatically after 1.5s (gives Flask time to start)
    threading.Timer(1.5, _open_browser).start()
    app.run(host="0.0.0.0", port=5000, debug=True)
