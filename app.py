# app.py - Main Flask application
import os
import logging
import tempfile
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
from ml_model import FakeReviewDetector

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "fake_review_detector_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Path to your CSV file
REVIEWS_CSV_PATH = 'data/reviews.csv'

# Initialize the ML model
review_detector = FakeReviewDetector()
review_detector.initialize_model(csv_path=REVIEWS_CSV_PATH)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/upload', methods=['GET'])
def upload_form():
    """Show the dataset upload form"""
    return render_template('upload.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """Process dataset upload and train model"""
    if 'file' not in request.files:
        return render_template('upload.html', error="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', error="No file selected")
    
    if not file.filename.endswith('.csv'):
        return render_template('upload.html', error="File must be a CSV file")
    
    try:
        # Create a temporary file to store the uploaded CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)
        
        # Re-initialize the model with the new dataset
        accuracy = review_detector.initialize_model(csv_path=temp_path)
        
        # Get some statistics
        X, y = review_detector.load_dataset_from_csv(temp_path)
        total = len(X)
        fake_count = sum(y)
        genuine_count = total - fake_count
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        success_message = f"Model successfully trained with an accuracy of {accuracy:.2%}!"
        stats = f"Total reviews: {total}, Fake: {fake_count}, Genuine: {genuine_count}"
        
        return render_template('upload.html', success=success_message, stats=stats)
    
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        return render_template('upload.html', error=f"Error processing CSV file: {str(e)}")

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a single review"""
    try:
        data = request.json
        review_text = data.get('review', '')
        
        if not review_text:
            return jsonify({'error': 'No review text provided'}), 400
        
        result = review_detector.analyze_review(review_text)
        logger.debug(f"Analysis result: {result}")
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error analyzing review: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    """Analyze multiple reviews"""
    try:
        data = request.json
        reviews = data.get('reviews', [])
        
        if not reviews:
            return jsonify({'error': 'No reviews provided'}), 400
        
        results = review_detector.analyze_batch(reviews)
        logger.debug(f"Batch analysis complete, {len(results)} reviews processed")
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
