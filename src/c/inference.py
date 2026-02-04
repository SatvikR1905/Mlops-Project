from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np

# ✅ Prometheus
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)
session = ort.InferenceSession("models/v1/model.onnx")

# Metrics
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
REQUEST_FAILS = Counter("inference_failures_total", "Total inference failures")
REQUEST_LATENCY = Histogram("inference_latency_seconds", "Inference latency in seconds")

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.inc()

    payload = request.get_json(silent=True)
    if not payload or "input" not in payload:
        REQUEST_FAILS.inc()
        return jsonify({"error": "JSON body must contain 'input'"}), 400

    arr = np.array(payload["input"], dtype=np.float32)

    # ✅ Ensure rank is 2: (batch, features)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    try:
        with REQUEST_LATENCY.time():
            outputs = session.run(None, {session.get_inputs()[0].name: arr})
        return jsonify({"predictions": outputs[0].tolist()})
    except Exception as e:
        REQUEST_FAILS.inc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
