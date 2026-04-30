# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI(title="MicroTurn AI Permanent Server")

# This allows Unity to talk to the server safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
print("Loading Models...")
rf_model = joblib.load('final_rf.pkl')
xgb_model = joblib.load('final_xgb.pkl')
scaler = joblib.load('final_scaler.pkl')

class MachiningInputs(BaseModel):
    vc: float
    doc: float
    feed: float
    time_cut: float

@app.post("/predict")
def predict_outcomes(data: MachiningInputs):
    input_df = pd.DataFrame({
        'Cutting Speed Vc (m/min)': [data.vc],
        'Feed f (mm/rev)':          [data.feed],
        'Depth of Cut (mm)':        [data.doc],
        'Machining Time (sec)':     [data.time_cut],
    })
    
    input_scaled = scaler.transform(input_df)
    preds = rf_model.predict(input_scaled)[0]
    risk = int(xgb_model.predict(input_scaled)[0])
    
    return {
        "Ra": round(preds[0], 3),
        "VB": round(preds[1], 2),
        "Load": round(preds[2], 1),
        "Vibration": round(preds[3], 2),
        "Risk_Level": risk
    }