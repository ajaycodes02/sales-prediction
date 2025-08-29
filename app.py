from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

model = joblib.load('sales_model.pkl')
scaler = joblib.load('scaler.pkl')

with open('feature_importance.json', 'r') as f:
    feature_importance = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']
        
        # Validate input
        required_keys = ['ad_spend', 'prev_sales', 'season']
        if not all(key in features for key in required_keys):
            return jsonify({'error': 'Missing required features'}), 400
        
        # Map season to numeric
        season_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
        if features['season'] not in season_map:
            return jsonify({'error': 'Invalid season value'}), 400
        features['season'] = season_map[features['season']]
        
        # Prepare input for model
        input_data = np.array([
            features['ad_spend'],
            features['prev_sales'],
            features['season']
        ]).reshape(1, -1)
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'feature_importance': feature_importance
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    
@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Sales Prediction API. Use POST /predict for predictions.'})