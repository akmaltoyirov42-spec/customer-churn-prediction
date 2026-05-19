# Customer Churn Prediction

![CI](https://github.com/akmaltoyirov42-spec/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)

Telecom customer churn predictor. I trained three models (Logistic Regression, Random Forest, XGBoost), compared them by ROC-AUC, and wrapped the winner in a REST API. XGBoost ended up the best — not surprising given the mix of categorical and numeric features here.

Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7 043 customers, 20 features.

---

## Results

| Model | ROC-AUC | F1 (churn class) |
|---|---|---|
| Logistic Regression | 0.843 | 0.61 |
| Random Forest | 0.851 | 0.62 |
| **XGBoost** | **0.867** | **0.65** |

The dataset is imbalanced (~26% churn). Accuracy would've been misleading, so I tracked ROC-AUC and F1 on the churn class specifically.

---

## How it works

```
raw CSV → clean + encode → train/test split → scale → fit 3 models
                                                              ↓
                                                    pick best by AUC
                                                              ↓
                                                    save model + scaler
                                                              ↓
                                                    FastAPI serves predictions
```

The API takes customer features as JSON and returns a churn probability plus a label.

---

## Quickstart

```bash
git clone https://github.com/akmaltoyirov42-spec/customer-churn-prediction.git
cd customer-churn-prediction

python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Download the dataset from Kaggle (see `data/README.md`), then:

```bash
# Train the model (takes ~30 seconds)
python -m src.train

# Start the API
uvicorn api.main:app --reload
```

API docs available at `http://localhost:8000/docs` once it's running.

---

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "MonthlyCharges": 85.0,
    "TotalCharges": 170.0,
    "Contract_One_year": 0,
    "Contract_Two_year": 0,
    "PaymentMethod_Electronic_check": 1,
    "InternetService_Fiber_optic": 1,
    "PaperlessBilling": 1
  }'
```

Response:
```json
{
  "churn_probability": 0.7841,
  "prediction": "churn"
}
```

---

## Docker

```bash
# Train first so the model exists
python -m src.train

docker build -t churn-api .
docker run -p 8000:8000 -v $(pwd)/model:/app/model churn-api
```

---

## Tests

```bash
pytest tests/ -v
```

Tests don't need a trained model — they mock the predictor and check the API contract (routes, schema, error codes). CI runs them on every push via GitHub Actions.

---

## Files

```
├── src/
│   ├── preprocess.py   data cleaning + encoding
│   ├── train.py        model training + comparison
│   └── predict.py      inference wrapper
├── api/
│   └── main.py         FastAPI app
├── tests/
│   └── test_api.py
├── .github/workflows/ci.yml
└── data/README.md      dataset download instructions
```

---

## Notes

- `TotalCharges` had blank strings instead of NaN — caught that during EDA and handled it in the preprocessing step
- Used `scale_pos_weight` in XGBoost to deal with class imbalance instead of oversampling, which I find cleaner
- The scaler is saved alongside the model so the API uses identical preprocessing to training
