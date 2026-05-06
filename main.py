from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the Champion XGBoost model and the Scaler
# Ensure these filenames match exactly what you uploaded to GitHub
model = joblib.load('machining_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from Unity
        data = request.json
        
        # 1. Arrange inputs in the exact order the model expects
        # [Speed, Feed, Depth, Time]
        input_data = pd.DataFrame([[
            float(data['speed']), 
            float(data['feed']), 
            float(data['depth']), 
            float(data['time'])
        ]], columns=['Cutting Speed Vc (m/min)', 'Feed f (mm/rev)', 'Depth of Cut (mm)', 'Machining Time (sec)'])
        
        # 2. Apply the Scaler (This makes the Unity input match the training data)
        scaled_input = scaler.transform(input_data)
        
        # 3. Generate Prediction
        prediction = model.predict(scaled_input)
        
        # 4. Extract Ra and Tool Wear
        # Note: prediction[0][0] is Ra, prediction[0][1] is VB (Wear)
        predicted_ra = float(prediction[0][0])
        predicted_vb = float(prediction[0][1])

        # 5. Return JSON to Unity
        return jsonify({
            "ra": round(predicted_ra, 3),
            "wear": round(predicted_vb, 1),
            "status": "Success"
        })

    except Exception as e:
        return jsonify({
            "status": "Error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    app.run()
