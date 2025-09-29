from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow import keras   # 딥러닝 모델 사용 시 필요
import os

app = Flask(__name__)

# === 모델 로드 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "models")

# Scikit-learn 모델
lr_model = joblib.load(os.path.join(models_dir, "logistic_regression_model.pkl"))
rf_model = joblib.load(os.path.join(models_dir, "random_forest_model.pkl"))

# 전처리기 & 팀 매핑
scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
team_to_id = joblib.load(os.path.join(models_dir, "team_mapping.pkl"))

# 딥러닝 모델 (있을 경우)
dl_model_path = os.path.join(models_dir, "deep_learning_model.h5")
dl_model = keras.models.load_model(dl_model_path) if os.path.exists(dl_model_path) else None


@app.route("/")
def home():
    return "Baseball Predictor API is running!"


# === 예측 API ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        avg = data.get("avg")
        era = data.get("era")
        runs = data.get("runs")
        model_type = data.get("model", "logistic")
        # "logistic", "random_forest", "deep_learning" 중 선택

        # 입력값 → numpy
        X = np.array([[avg, era, runs]])
        X_scaled = scaler.transform(X)

        # === 모델 선택 ===
        if model_type == "logistic":
            prob = lr_model.predict_proba(X_scaled)[0][1]
        elif model_type == "random_forest":
            prob = rf_model.predict_proba(X_scaled)[0][1]
        elif model_type == "deep_learning" and dl_model is not None:
            prob = float(dl_model.predict(X_scaled)[0][0])  # sigmoid 출력 가정
        else:
            return jsonify({"error": f"지원하지 않는 모델 타입: {model_type}"}), 400

        return jsonify({
            "model": model_type,
            "win_probability": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
