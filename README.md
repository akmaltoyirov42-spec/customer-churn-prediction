# customer churn prediction

![CI](https://github.com/akmaltoyirov42-spec/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)

predicts whether a telecom customer will leave. tried 3 different models, picked the best, then wrapped it in a FastAPI endpoint so it actually feels like a real ML project.

dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7k customers, 20 features.

---

## results

| model | ROC-AUC | F1 (churn) |
|---|---|---|
| Logistic Regression | 0.843 | 0.61 |
| Random Forest | 0.851 | 0.62 |
| **XGBoost** | **0.867** | **0.65** |

i learned the hard way that accuracy is the wrong metric here — only 26% of customers churn, so a model that always predicts "no churn" gets 74% accuracy and is completely useless. ROC-AUC is way better for this.

---

## run it

```bash
git clone https://github.com/akmaltoyirov42-spec/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt

# grab the dataset from kaggle first (link above), then:
python -m src.train

uvicorn api.main:app --reload
# docs at http://localhost:8000/docs
```

## example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 85.0, "TotalCharges": 170.0, "Contract_One_year": 0, "PaymentMethod_Electronic_check": 1}'
```

```json
{"churn_probability": 0.7841, "prediction": "churn"}
```

## docker

```bash
python -m src.train
docker build -t churn-api .
docker run -p 8000:8000 -v $(pwd)/model:/app/model churn-api
```

## tests

```bash
pytest tests/ -v
```

tests mock the model so CI doesn't need a trained file.

---

## things i ran into

- `TotalCharges` column had blank strings instead of NaN — pandas didn't catch them as missing, had to fix in preprocessing
- used `scale_pos_weight` in XGBoost for the class imbalance instead of oversampling (simpler and worked just as well)
- saving the scaler alongside the model is important — otherwise inference uses different preprocessing than training

---

pandas, scikit-learn, XGBoost, FastAPI, Docker, GitHub Actions
