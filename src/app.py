import os
import json
from typing import List
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import joblib

# --- 1. Setup & Configuration ---
LOG_FILE = Path("prediction_log.jsonl")
app = FastAPI(title="HVAC Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 2. WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# --- 3. Model Loading (STRICT - No Mocks) ---

def load_real_models():
    """
    Attempts to locate and load models from likely directories.
    """
    # List of places to look for the 'models' folder or the files directly
    possible_paths = [
        os.path.join(BASE_DIR, 'models'),       # Subfolder: ./models/
        os.path.join(BASE_DIR, '..', 'models'), # Parent sibling: ../models/
        BASE_DIR,                               # Current folder: ./
        '/app/models'                           # Docker standard path
    ]
    
    found_path = None
    
    # Check where the classifier file actually exists
    print("--- Searching for models ---")
    for path in possible_paths:
        test_file = os.path.join(path, 'best_rf_classifier.pkl')
        if os.path.exists(test_file):
            found_path = path
            print(f"✅ Found models in: {found_path}")
            break
        else:
            print(f"❌ Not found in: {path}")

    if not found_path:
        raise FileNotFoundError(
            "CRITICAL ERROR: Could not find 'best_rf_classifier.pkl' in any standard path. "
            "Please check your folder structure."
        )

    # Load the files
    try:
        clf = joblib.load(os.path.join(found_path, 'best_rf_classifier.pkl'))
        reg = joblib.load(os.path.join(found_path, 'best_rf_regressor.pkl'))
        sc = joblib.load(os.path.join(found_path, 'scaler.pkl'))
        print("✅ All models loaded successfully!")
        return clf, reg, sc
    except Exception as e:
        print(f"CRITICAL ERROR: Files found but failed to load. Reason: {e}")
        # Common error: sklearn version mismatch
        raise e

# EXECUTE LOADING
# If this fails, the server will crash immediately (which is good for debugging)
clf_model, reg_model, scaler = load_real_models()


# --- 4. Pipeline Definition ---
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

pipeline = Pipeline([('scaler', scaler), ('hvac_model', HVACModel(reg_model, clf_model))])

# --- 5. Data Structures ---
class InputData(BaseModel):
    out_temp: float
    out_hum: float
    num_people: int
    room_area: float
    active_units: int
    hour: int
    day: int
    prev_setpoint: float
    prev_fan_speed: int
    prev_out_temp: float
    room_name: str

class BatchInput(BaseModel):
    data: List[InputData]

def log_request(input_data, prediction):
    with LOG_FILE.open("a") as f:
        f.write(json.dumps({"input": input_data, "prediction": prediction}) + "\n")

# --- 6. Routes ---

@app.get("/")
async def get_dashboard():
    html_path = os.path.join(BASE_DIR, "dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    return HTMLResponse(content="<h1>dashboard.html not found. Check file location.</h1>", status_code=404)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/predict")
async def predict_batch(batch: BatchInput):
    try:
        records = [item.model_dump() for item in batch.data]
        df = pd.DataFrame(records)

        required_columns = [
            'out_temp', 'out_hum', 'num_people', 'room_area', 'active_units',
            'hour', 'day', 'prev_setpoint', 'prev_fan_speed', 'prev_out_temp', 'room_name'
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

        X = df[required_columns].drop(columns=['room_name'])

        # Prediction
        fan_speed_preds, setpoint_preds = pipeline.predict(X)

        results = [
            {   "room_name": record['room_name'],
                "predicted_setpoint": float(setpoint),
                "predicted_fan_speed": int(fan_speed),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            for setpoint, fan_speed, record in zip(setpoint_preds, fan_speed_preds, records)
        ]

        # log_request(records, results)

        await manager.broadcast({
            "type": "prediction_update",
            "inputs": records,
            "outputs": results
        })

        return {"predictions": results}

    except Exception as e:
        # Detailed error printing
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_single")
async def predict_single(data: InputData):
    return await predict_batch(BatchInput(data=[data]))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)