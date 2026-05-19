from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict

app = FastAPI(
    title="Churn Prediction API",
    description="Returns churn probability for a telecom customer.",
    version="1.0.0",
)


class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, example=12)
    MonthlyCharges: float = Field(..., ge=0, example=65.5)
    TotalCharges: float = Field(..., ge=0, example=786.0)
    Contract_One_year: int = Field(0, ge=0, le=1)
    Contract_Two_year: int = Field(0, ge=0, le=1)
    PaymentMethod_Credit_card_automatic: int = Field(0, ge=0, le=1)
    PaymentMethod_Electronic_check: int = Field(0, ge=0, le=1)
    PaymentMethod_Mailed_check: int = Field(0, ge=0, le=1)
    InternetService_Fiber_optic: int = Field(0, ge=0, le=1)
    InternetService_No: int = Field(0, ge=0, le=1)
    OnlineSecurity_No_internet_service: int = Field(0, ge=0, le=1)
    OnlineSecurity_Yes: int = Field(0, ge=0, le=1)
    TechSupport_No_internet_service: int = Field(0, ge=0, le=1)
    TechSupport_Yes: int = Field(0, ge=0, le=1)
    PaperlessBilling: int = Field(0, ge=0, le=1)
    SeniorCitizen: int = Field(0, ge=0, le=1)
    Partner: int = Field(0, ge=0, le=1)
    Dependents: int = Field(0, ge=0, le=1)

    class Config:
        extra = "allow"  # forward-compatible with more features


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def get_prediction(customer: CustomerFeatures):
    try:
        result = predict(customer.model_dump())
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run `python -m src.train` first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
