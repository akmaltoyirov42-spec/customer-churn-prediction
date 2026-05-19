# Customer Churn Prediction

![CI](https://github.com/akmaltoyirov42-spec/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)

Predicts whether a telecom customer will churn. Trained three models, picked the best one by ROC-AUC, wrapped it in a FastAPI endpoint.

Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7k customers, 20 features.

---

## Results

| Model | ROC-AUC | F1 (churn) |
|---|---|---|
| Logistic Regression | 0.843 | 0.61 |
| Random Forest | 0.851 | 0.62 |
| **XGBoost** | **0.867** | **0.65** |

Used ROC-AUC instead of accuracy because only 26% of customers actually churn — accuracy alone is misleading on imbalanced data.

---

## Quickstart

```bash
git clone https://github.com/akmaltoyirov42-spec/customer-churn-prediction.git
cd customer-churn-prediction

pip install -r requirements.txt

# download dataset from Kaggle (see data/README.md), then:
python -m src.train

uvicorn api.main:app --reload
# docs at http://localhost:8000/docs
```

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 85.0, "TotalCharges": 170.0, "Contract_One_year": 0, "PaymentMethod_Electronic_check": 1}'
```

```json
{"churn_probability": 0.7841, "prediction": "churn"}
```

## Docker

```bash
python -m src.train
docker build -t churn-api .
docker run -p 8000:8000 -v $(pwd)/model:/app/model churn-api
```

## Tests

```bash
pytest tests/ -v
```

Tests mock the model so they run in CI without needing a trained file.

---

## Notes

- `TotalCharges` had blank strings instead of NaN — fixed in preprocessing
- Used `scale_pos_weight` in XGBoost for class imbalance instead of oversampling
- Scaler is saved alongside the model so inference uses the same preprocessing as training

---

## Stack

pandas, scikit-learn, XGBoost, FastAPI, Docker, GitHub Actions
