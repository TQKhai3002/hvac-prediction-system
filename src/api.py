import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import joblib

app = FastAPI(title="HVAC Prediction API")

# Dynamic model path
IN_DOCKER = os.getenv('IN_DOCKER', 'false').lower() == 'true'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = '/app/models' if IN_DOCKER else os.path.join(BASE_DIR, '..', 'models')

# Load models
try:
    print(f"Loading models from {MODELS_DIR}")
    clf_model = joblib.load(os.path.join(MODELS_DIR, 'best_rf_classifier.pkl'))
    reg_model = joblib.load(os.path.join(MODELS_DIR, 'best_rf_regressor.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
except FileNotFoundError as e:
    print(f"Error: Model file not found at {MODELS_DIR}: {e}")
    raise

class HVACModel(BaseEstimator):
    def __init__(self, reg_model, clf_model):
        self.reg_model = reg_model
        self.clf_model = clf_model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        clf_pred = self.clf_model.predict(X)
        reg_pred = self.reg_model.predict(X)
        return clf_pred, reg_pred

    def __sklearn_is_fitted__(self):
        return True

pipeline = Pipeline([('scaler', scaler), ('hvac_model', HVACModel(reg_model, clf_model))])

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
        required_columns = ['out_temp', 'out_hum', 'num_people', 'room_area', 'active_units', 'hour', 'day', 'prev_setpoint', 'prev_fan_speed', 'prev_out_temp']
        X = df[required_columns]
        fan_speed_pred, setpoint_pred = pipeline.predict(X)
        return {"predicted_setpoint": float(setpoint_pred[0]), "predicted_fan_speed": float(fan_speed_pred[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)