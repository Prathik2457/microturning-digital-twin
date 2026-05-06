from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# DEBUG: Print what files Render can actually see
print("Files in directory:", os.listdir())

try:
    model = joblib.load('machining_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and Scaler loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR LOADING FILES: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([[
            float(data['speed']), 
            float(data['feed']), 
            float(data['depth']), 
            float(data['time'])
        ]], columns=['Cutting Speed Vc (m/min)', 'Feed f (mm/rev)', 'Depth of Cut (mm)', 'Machining Time (sec)'])
        
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        
        return jsonify({
            "ra": float(round(prediction[0][0], 3)),
            "wear": float(round(prediction[0][1], 1)),
            "status": "Success"
        })
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run()
