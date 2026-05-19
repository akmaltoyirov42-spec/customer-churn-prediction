"""
Train three models and pick the best one by ROC-AUC.
Accuracy alone is misleading on churn data because only ~26% of customers actually churn.
"""

import joblib
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.preprocess import encode, load_data, scale, split_features

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = Path("model")


def evaluate(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n{name}  |  ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    return {"name": name, "auc": auc, "report": report, "model": model}


def train():
    df = load_data(DATA_PATH)
    df = encode(df)
    X, y = split_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_s, X_test_s, scaler = scale(X_train, X_test)

    candidates = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), True),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), False),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                # class_weight equiv in XGB — churn is ~26% so we push it a bit
                scale_pos_weight=int((y_train == 0).sum() / (y_train == 1).sum()),
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            ),
            False,
        ),
    ]

    results = []
    for name, model, needs_scaled in candidates:
        X_tr = X_train_s if needs_scaled else X_train
        X_te = X_test_s if needs_scaled else X_test
        model.fit(X_tr, y_train)
        results.append(evaluate(model, X_te, y_test, name))

    best = max(results, key=lambda r: r["auc"])
    print(f"\nBest model: {best['name']}  (AUC {best['auc']:.4f})")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best["model"], MODEL_DIR / "model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_names.pkl")

    metrics = {
        "best_model": best["name"],
        "roc_auc": round(best["auc"], 4),
        "classification_report": best["report"],
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {MODEL_DIR}/")
    return metrics


if __name__ == "__main__":
    train()
