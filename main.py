# main.py - Entry point for the Flask application
import os
from app import app  # noqa: F401

# Set path to CSV file
os.environ['REVIEWS_CSV_PATH'] = 'data/reviews.csv'

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5008, debug=True)
