from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pickle
import json
import numpy as np
import pandas as pd
import os
from werkzeug.exceptions import BadRequest
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Global variables to store model and columns
model = None
data_columns = None
locations = None

def load_saved_artifacts():
    """Load the trained model and column information"""
    global model, data_columns, locations
    
    try:
        # Load the trained model
        with open('banglor_home_prices_model.pickle', 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        
        # Load columns information
        with open('columns.json', 'r') as f:
            data_columns = json.load(f)['data_columns']
        
        # Extract location columns (all columns except the first 3: sqft, bath, bhk)
        locations = data_columns[3:]  # Skip total_sqft, bath, bhk
        logger.info(f"Loaded {len(locations)} locations")
        
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        raise Exception("Model files not found. Please ensure 'banglor_home_prices_model.pickle' and 'columns.json' are in the current directory.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise Exception(f"Error loading model: {e}")

def get_estimated_price(location, sqft, bhk, bath):
    """
    Predict house price using the trained model
    
    Args:
        location (str): Location name
        sqft (float): Total square feet
        bhk (int): Number of bedrooms
        bath (int): Number of bathrooms
    
    Returns:
        float: Predicted price in lakhs
    """
    try:
        # Create input array with zeros
        x = np.zeros(len(data_columns))
        
        # Set basic features
        x[0] = sqft      # total_sqft
        x[1] = bath      # bath  
        x[2] = bhk       # bhk
        
        # Set location feature if it exists in the model
        location_lower = location.lower().strip()
        
        # Try to find matching location
        location_index = None
        for i, loc in enumerate(locations):
            if loc.lower() == location_lower:
                location_index = i + 3  # +3 because locations start after sqft, bath, bhk
                break
        
        if location_index is not None:
            x[location_index] = 1
            logger.info(f"Location '{location}' found at index {location_index}")
        else:
            logger.warning(f"Location '{location}' not found in model. Using 'other' category.")
        
        # Make prediction
        prediction = model.predict([x])[0]
        
        # Round to 2 decimal places
        return round(prediction, 2)
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise Exception(f"Error in prediction: {e}")

@app.route('/')
def home():
    """Serve the prediction interface directly"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>❌ Interface Not Found</h1><p>Please place index.html in your project folder.</p>"


# ADD THIS NEW ROUTE HERE
@app.route('/app')
def serve_app():
    """Serve the price prediction interface"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return render_template_string("""
        <div style="text-align: center; padding: 50px; font-family: Arial;">
            <h1>❌ Interface Not Found</h1>
            <p>The index.html file is missing from your Flask app directory.</p>
            <p>Please save the HTML interface file as 'index.html' in the same folder as your Flask app.</p>
            <a href="/" style="color: #007bff;">← Back to API Documentation</a>
        </div>
        """)

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    """Return all available location names"""
    try:
        if locations is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Convert to title case for better display
        formatted_locations = [loc.replace('_', ' ').title() for loc in locations]
        
        return jsonify({
            'locations': formatted_locations,
            'total_count': len(locations)
        })
        
    except Exception as e:
        logger.error(f"Error getting locations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    """Predict home price based on input parameters"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Extract parameters
        location = data.get('location', '').strip()
        sqft = data.get('total_sqft')
        bhk = data.get('bhk')
        bath = data.get('bath')
        
        # Validate required parameters
        if not all([location, sqft is not None, bhk is not None, bath is not None]):
            return jsonify({
                'error': 'Missing required parameters: location, total_sqft, bhk, bath'
            }), 400
        
        # Validate data types and ranges
        try:
            sqft = float(sqft)
            bhk = int(bhk)
            bath = int(bath)
        except (ValueError, TypeError):
            return jsonify({
                'error': 'Invalid data types. sqft should be number, bhk and bath should be integers'
            }), 400
        
        # Validate ranges
        if sqft < 300 or sqft > 20000:
            return jsonify({'error': 'Total square feet should be between 300 and 20000'}), 400
        if bhk < 1 or bhk > 10:
            return jsonify({'error': 'BHK should be between 1 and 10'}), 400
        if bath < 1 or bath > 15:
            return jsonify({'error': 'Bathrooms should be between 1 and 15'}), 400
        if bath > bhk + 3:
            return jsonify({'error': 'Number of bathrooms seems unrealistic for the given BHK'}), 400
        if sqft / bhk < 200:
            return jsonify({'error': 'Square feet per BHK seems too low (minimum 200 sq ft per BHK)'}), 400
        
        # Make prediction
        estimated_price = get_estimated_price(location, sqft, bhk, bath)
        
        return jsonify({
            'estimated_price': estimated_price,
            'location': location,
            'total_sqft': sqft,
            'bhk': bhk,
            'bath': bath,
            'price_per_sqft': round((estimated_price * 100000) / sqft, 2),
            'currency': 'INR Lakhs'
        })
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'model_loaded': model is not None,
            'locations_loaded': locations is not None,
            'total_locations': len(locations) if locations else 0
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get detailed model information"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        info = {
            'model_type': str(type(model).__name__),
            'feature_count': len(data_columns) if data_columns else 0,
            'location_count': len(locations) if locations else 0,
            'features': {
                'numerical': ['total_sqft', 'bath', 'bhk'],
                'categorical': ['location']
            },
            'model_params': {
                'intercept': float(model.intercept_) if hasattr(model, 'intercept_') else None,
                'n_features': int(model.n_features_in_) if hasattr(model, 'n_features_in_') else None
            }
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        # Load model and data when starting the server
        load_saved_artifacts()
        logger.info("Starting Flask server...")
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"Error: {e}")
        print("\nPlease ensure you have the following files in your current directory:")
        print("- banglor_home_prices_model.pickle")
        print("- columns.json")