from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow import keras
import os
from ensemble import CustomEnsemble


app = Flask(__name__)

# === 모델 로드 헬퍼 함수 ===
def load_model_safely(path, loader, name):
    try:
        if os.path.exists(path):
            return loader(path)
        else:
            app.logger.warning(f"[WARN] {name} 파일이 존재하지 않습니다: {path}")
            return None
    except Exception as e:
        app.logger.error(f"[ERROR] {name} 로딩 실패: {e}")
        return None


ensemble_model = joblib.load("models/ensemble_model.pkl")



@app.route("/")
def home():
    return "Baseball Predictor API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        home_ops = float(data.get("home_ops"))
        home_era = float(data.get("home_era"))
        away_ops = float(data.get("away_ops"))
        away_era = float(data.get("away_era"))
        temperature = float(data.get("temperature", 25))
        humidity = float(data.get("humidity", 70))

        # 예측 실행
        result = ensemble_model.predict(
            home_ops, home_era, away_ops, away_era, temperature, humidity
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
