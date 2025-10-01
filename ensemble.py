# ensemble.py
import numpy as np

class CustomEnsemble:
    def __init__(self, lr_model, rf_model, dl_model=None, scaler=None, team_mapping=None):
        self.lr_model = lr_model
        self.rf_model = rf_model
        self.dl_model = dl_model
        self.scaler = scaler
        self.team_mapping = team_mapping

    def predict(self, home_ops, home_era, away_ops, away_era, temperature=25, humidity=70):
        ops_diff = home_ops - away_ops
        era_diff = away_era - home_era
        features = np.array([[home_ops, home_era, away_ops, away_era,
                              temperature, humidity, ops_diff, era_diff]])
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        lr_prob = self.lr_model.predict_proba(features_scaled)[0][1]
        rf_prob = self.rf_model.predict_proba(features_scaled)[0][1]
        if self.dl_model is not None:
            dl_prob = float(self.dl_model.predict(features_scaled)[0][0])
            ensemble_prob = (lr_prob * 0.3 + rf_prob * 0.4 + dl_prob * 0.3)
            individual_predictions = {
                'LogisticRegression': float(lr_prob),
                'RandomForest': float(rf_prob),
                'DeepLearning': float(dl_prob)
            }
        else:
            ensemble_prob = (lr_prob * 0.4 + rf_prob * 0.6)
            individual_predictions = {
                'LogisticRegression': float(lr_prob),
                'RandomForest': float(rf_prob),
                'DeepLearning': 'N/A'
            }
        return {
            'home_win_probability': float(ensemble_prob),
            'away_win_probability': float(1 - ensemble_prob),
            'predicted_winner': '홈팀' if ensemble_prob > 0.5 else '원정팀',
            'confidence': float(max(ensemble_prob, 1 - ensemble_prob)),
            'individual_predictions': individual_predictions
        }
