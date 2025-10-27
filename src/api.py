from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import joblib

app = FastAPI(title="HVAC Prediction API")

# Load models (Docker path)
clf_model = joblib.load('/app/models/best_rf_classifier.pkl')
reg_model = joblib.load('/app/models/best_rf_model.pkl')
scaler = joblib.load('/app/models/scaler.pkl')

# Pipeline (same as consumer)

class InputData(BaseModel):
    out_temp: float
    out_hum: float
    num_people: int
    room_area: float
    active_units: int
    hour: int
    day: int

@app.post("/predict")
async def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])
        required_columns = ['out_temp', 'out_hum', 'num_people', 'room_area', 'active_units', 'hour', 'day']
        X = df[required_columns]
        fan_speed_pred, setpoint_pred = pipeline.predict(X)
        return {"predicted_setpoint": float(setpoint_pred[0]), "predicted_fan_speed": float(fan_speed_pred[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
