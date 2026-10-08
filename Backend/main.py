from fastapi import FastAPI
from pydantic import BaseModel

from Backend.predictor import predict_insurance_charges

app = FastAPI(
    title="Medical Insurance Charges Prediction API",
    description="This API predicts medical insurance charges based on user input features.",
    version="1.0.0"
)

#input schema matching features used in training
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

#medical insurance charges prediction endpoint
@app.post("/predict")
def predict_charges(input_data: InsuranceInput):
    input_data=input_data.model_dump()  # Convert Pydantic model to dictionary

    result = predict_insurance_charges(input_data)
    return {"predicted_charges": round(result, 2)}