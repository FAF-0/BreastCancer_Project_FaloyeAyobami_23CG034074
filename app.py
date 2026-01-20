from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and preprocessing objects
MODEL_PATH = 'model/breast_cancer_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
ENCODER_PATH = 'model/label_encoder.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.json
        
        # Extract features
        features = [
            float(data['radius_mean']),
            float(data['texture_mean']),
            float(data['perimeter_mean']),
            float(data['area_mean']),
            float(data['smoothness_mean'])
        ]
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_label = encoder.inverse_transform([prediction])[0]
        
        # Get prediction probability
        if hasattr(model, 'decision_function'):
            confidence = abs(model.decision_function(features_scaled)[0])
        else:
            confidence = 0.5
        
        return jsonify({
            'success': True,
            'diagnosis': prediction_label,
            'diagnosis_full': 'Malignant (M)' if prediction_label == 'M' else 'Benign (B)',
            'confidence': min(round(confidence, 2), 1.0)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/info', methods=['GET'])
def info():
    """Provide model information"""
    return jsonify({
        'model': 'Support Vector Machine (SVM)',
        'features': [
            'radius_mean',
            'texture_mean', 
            'perimeter_mean',
            'area_mean',
            'smoothness_mean'
        ],
        'classes': ['Benign (B)', 'Malignant (M)'],
        'disclaimer': 'For educational purposes only. Not a medical diagnostic tool.'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)