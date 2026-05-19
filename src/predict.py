import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("model")

_model = None
_scaler = None
_feature_names = None


def _load():
    global _model, _scaler, _feature_names
    if _model is None:
        _model = joblib.load(MODEL_DIR / "model.pkl")
        _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        _feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")


def predict(features: dict) -> dict:
    _load()
    df = pd.DataFrame([features])
    df = df.reindex(columns=_feature_names, fill_value=0)

    # scaler only used for LR; tree models ignore it but it doesn't hurt
    X = _scaler.transform(df)
    prob = float(_model.predict_proba(X)[0][1])
    label = "churn" if prob >= 0.5 else "no churn"

    return {"churn_probability": round(prob, 4), "prediction": label}
