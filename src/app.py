import os
import sys
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fight_predictor import UFCFightPredictor

app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
    "expose_headers": ["Content-Length", "Content-Type"],
    "supports_credentials": True,
    "max_age": 86400
}})

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Length, Content-Type'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response

# Handle preflight requests
@app.route('/predict', methods=['OPTIONS'])
def handle_preflight():
    response = jsonify({})
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    return response

# Initialize the predictor globally
try:
    predictor = UFCFightPredictor()
    logger.info("Successfully initialized UFCFightPredictor")
    logger.info(f"Available fighters: {predictor.get_fighter_names()}")
except Exception as e:
    logger.error(f"Failed to initialize UFCFightPredictor: {str(e)}", exc_info=True)
    raise

app.config['TEMPLATES_AUTO_RELOAD'] = True

# Adjust paths to point to the project root
project_root = os.path.dirname(os.path.abspath(__file__))
app.config['STATIC_FOLDER'] = os.path.join(os.path.dirname(project_root), 'static')
app.config['TEMPLATES_FOLDER'] = os.path.join(os.path.dirname(project_root), 'templates')
app.config['PROPAGATE_EXCEPTIONS'] = True

@app.route('/')
def index():
    try:
        sys.stdout.write("APP DEBUG: Index route accessed\n")
        # Get all fighters
        fighters = []
        fighter_names = predictor.get_fighter_names()
        if not fighter_names:
            sys.stdout.write("APP DEBUG: No fighters found in database\n")
            return render_template('error.html', error="No fighters found in database"), 500
        
        for fighter_name in fighter_names:
            fighter_data = predictor.get_fighter_data(fighter_name)
            if fighter_data:
                sys.stdout.write(f"APP DEBUG: Fighter data for {fighter_name}: {fighter_data}\n")
                fighters.append({
                    'name': fighter_name,
                    'image_path': fighter_data.get('image_path', 'static/images/default_fighter.jpg')
                })
        return render_template('index.html', fighters=fighters)
    except Exception as e:
        sys.stdout.write(f"APP DEBUG: Error in index route: {str(e)}\n")
        return render_template('error.html', error=str(e)), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log incoming request
        logger.debug(f"Prediction request received: {request.json}")
        logger.debug(f"Raw request data: {request.data}")
        logger.debug(f"Request headers: {dict(request.headers)}")
        logger.debug(f"Request method: {request.method}")
        logger.debug(f"Request URL: {request.url}")
        logger.debug(f"Request content type: {request.headers.get('Content-Type')}")
        
        # Get request data
        data = request.get_json(force=True, silent=True)
        if not data:
            logger.error("No JSON data in request")
            logger.error(f"Raw request data: {request.data}")
            logger.error(f"Request content type: {request.headers.get('Content-Type')}")
            logger.error(f"Request headers: {dict(request.headers)}")
            return jsonify({
                'error': 'No data received in request',
                'details': 'Request body is empty or not JSON',
                'received_headers': dict(request.headers),
                'received_data': request.data.decode('utf-8') if request.data else None,
                'request_method': request.method,
                'request_url': request.url,
                'content_type': request.headers.get('Content-Type')
            }), 400
            
        # Validate request data
        if not isinstance(data, dict):
            logger.error(f"Invalid request data format: {type(data)}")
            logger.error(f"Raw data: {data}")
            return jsonify({
                'error': 'Invalid request data format',
                'details': 'Expected JSON object',
                'received_data': str(data),
                'data_type': str(type(data))
            }), 400
            
        # Get fighters
        fighter1 = data.get('fighter1')
        fighter2 = data.get('fighter2')
        
        if not fighter1 or not fighter2:
            logger.error(f"Invalid fighters: {fighter1}, {fighter2}")
            logger.error(f"Received data: {data}")
            return jsonify({
                'error': 'Both fighters must be selected',
                'received_data': data,
                'details': 'Missing fighter1 or fighter2 in request',
                'available_fighters': predictor.get_fighter_names()
            }), 400
            
        # Log fighters being processed
        logger.debug(f"Processing prediction for fighters: {fighter1}, {fighter2}")
        logger.debug(f"Request data: {data}")
        
        # Get prediction
        try:
            prediction = predictor.predict_fight(fighter1, fighter2)
            
            # Check if prediction returned an error
            if isinstance(prediction, dict) and 'error' in prediction:
                logger.error(f"Prediction error: {prediction['error']}")
                logger.error(f"Prediction response: {prediction}")
                return jsonify({
                    'error': prediction['error'],
                    'details': 'Prediction failed in fight_predictor',
                    'received_data': data
                }), 400
                
            # Format response
            response = {
                'winner': prediction['winner'],
                'method': prediction['method'],
                'round': int(prediction['round']),
                'confidence': prediction['confidence']
            }
            logger.debug(f"Prediction successful: {response}")
            return jsonify(response)
            
        except Exception as pred_error:
            logger.error(f"Prediction failed with error: {str(pred_error)}", exc_info=True)
            logger.error(f"Failed with data: {data}")
            return jsonify({
                'error': f'Prediction failed: {str(pred_error)}',
                'fighters': {'fighter1': fighter1, 'fighter2': fighter2},
                'details': 'Error during prediction processing',
                'received_data': data
            }), 400
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        logger.error(f"Failed with request: {request}")
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'details': 'Internal server error',
            'received_data': request.data.decode('utf-8') if request.data else None,
            'request_method': request.method,
            'request_url': request.url,
            'content_type': request.headers.get('Content-Type')
        }), 500

if __name__ == '__main__':
    try:
        print("Starting UFC Fight Predictor...")
        app.run(debug=True)
    except Exception as e:
        print(f"Error starting app: {str(e)}")