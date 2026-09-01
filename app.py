"""
app.py — local Flask server for the heart disease risk frontend.

Run:
    python app.py
Then open http://localhost:5000
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from train_model import build_features

app = Flask(__name__)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
COLUMNS_PATH = "feature_columns.json"

model = None
scaler = None
feature_columns = None


def load_artifacts():
    global model, scaler, feature_columns
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(COLUMNS_PATH)):
        return False
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(COLUMNS_PATH) as f:
        feature_columns = json.load(f)
    return True


ARTIFACTS_READY = load_artifacts()


@app.route("/")
def index():
    return render_template("index.html", ready=ARTIFACTS_READY)


@app.route("/predict", methods=["POST"])
def predict():
    if not ARTIFACTS_READY:
        return jsonify(
            {"error": "Model not trained yet. Run `python train_model.py` first (needs heart.csv in this folder)."}
        ), 400

    payload = request.get_json(force=True)

    required = [
        "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
        "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
    ]
    missing = [f for f in required if f not in payload or payload[f] in ("", None)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        row = {
            "Age": float(payload["Age"]),
            "Sex": payload["Sex"],
            "ChestPainType": payload["ChestPainType"],
            "RestingBP": float(payload["RestingBP"]),
            "Cholesterol": float(payload["Cholesterol"]),
            "FastingBS": int(payload["FastingBS"]),
            "RestingECG": payload["RestingECG"],
            "MaxHR": float(payload["MaxHR"]),
            "ExerciseAngina": payload["ExerciseAngina"],
            "Oldpeak": float(payload["Oldpeak"]),
            "ST_Slope": payload["ST_Slope"],
        }
    except (ValueError, TypeError):
        return jsonify({"error": "One or more numeric fields contain an invalid value."}), 400

    df = pd.DataFrame([row])
    processed = build_features(df)

    # Align to the exact training column set/order, filling any dummy
    # category the model has never seen with 0.
    processed = processed.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(processed)
    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][1])

    return jsonify(
        {
            "prediction": prediction,
            "label": "Higher likelihood of heart disease" if prediction == 1 else "Lower likelihood of heart disease",
            "probability": round(probability, 4),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
