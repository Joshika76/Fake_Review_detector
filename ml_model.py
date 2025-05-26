# ml_model.py - Machine learning model for fake review detection
import logging
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import string
import pickle
import os
from utils import generate_explanation, extract_features

# Configure logging
logger = logging.getLogger(__name__)

class FakeReviewDetector:
    """Class for detecting fake reviews using NLP and ML techniques"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = None
        
        # Download required NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Simple tokenization (avoid NLTK tokenizer issues)
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(processed_tokens)
    
    def load_dataset_from_csv(self, csv_path):
        """
        Load a dataset from a CSV file
        
        Expected CSV format:
        - A column containing the review text
        - A column indicating whether the review is fake (1) or genuine (0)
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            X: List of review texts
            y: List of labels (1 for fake, 0 for genuine)
        """
        try:
            # Load the dataset
            logger.info(f"Loading dataset from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # For the specific dataset provided (reviews.csv)
            if 'label' in df.columns and 'text' in df.columns:
                text_column = 'text'
                label_column = 'label'
                
                # Extract data
                X = df[text_column].tolist()
                
                # Convert labels to 0/1 format
                # In this dataset, 'CG' means 'computer-generated' (fake) and 'OR' means 'human-written' (genuine)
                y = [1 if str(label) == 'CG' else 0 for label in df[label_column]]
                
                logger.info(f"Loaded {len(X)} reviews with {sum(y)} fake and {len(y) - sum(y)} genuine")
                return X, y
            
            # Fallback to automatic detection for other datasets
            text_column = None
            label_column = None
            
            for col in df.columns:
                col_lower = col.lower()
                # Try to find the text column
                if 'text' in col_lower or 'review' in col_lower or 'content' in col_lower or 'comment' in col_lower:
                    text_column = col
                
                # Try to find the label column
                if 'label' in col_lower or 'fake' in col_lower or 'genuine' in col_lower or 'class' in col_lower:
                    label_column = col
            
            if not text_column or not label_column:
                logger.warning("Could not automatically detect column names. Using first two columns.")
                text_column = df.columns[0]
                label_column = df.columns[1]
            
            logger.info(f"Using '{text_column}' as text column and '{label_column}' as label column")
            
            # Extract data
            X = df[text_column].tolist()
            
            # Convert labels to 0/1 format if needed
            if df[label_column].dtype == 'object':
                # If labels are strings, convert to 0/1
                # Assuming "fake", "true", "1", "yes" etc. indicate fake reviews
                fake_indicators = ['fake', 'yes', '1', 'true', 't', 'y', 'deceptive', 'cg']
                y = [1 if str(label).lower() in fake_indicators else 0 for label in df[label_column]]
            else:
                # If labels are already numeric
                y = df[label_column].tolist()
            
            logger.info(f"Loaded {len(X)} reviews with {sum(y)} fake and {len(y) - sum(y)} genuine")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            # Fall back to synthetic data if there's an error
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for model training if no CSV is provided"""
        logger.info("Generating synthetic training data")
        np.random.seed(42)
        
        # Create sample reviews
        fake_patterns = [
            "This product is amazing! Best thing ever! Life changing!",
            "I can't believe how good this is! Absolutely perfect in every way!",
            "Wow, just wow! This exceeded all my expectations!",
            "Definitely a must buy! You won't regret it!",
            "I've tried many products but this is by far the best one!",
        ]
        
        genuine_patterns = [
            "I found this product to be useful. It has some flaws but overall good.",
            "Works as described. Shipping was a bit slow.",
            "I've been using this for a month now. The quality is decent.",
            "Good value for the price. I would recommend it with some caveats.",
            "It mostly meets my needs. Customer service was helpful when I had questions.",
        ]
        
        # Generate variations of these patterns
        fake_reviews = []
        genuine_reviews = []
        
        for pattern in fake_patterns:
            words = pattern.split()
            for _ in range(100):
                # Create variations by shuffling, dropping, or repeating words
                new_review = ' '.join(np.random.choice(words, size=len(words), replace=True))
                fake_reviews.append(new_review)
        
        for pattern in genuine_patterns:
            words = pattern.split()
            for _ in range(100):
                # Create variations by shuffling, dropping, or repeating words
                new_review = ' '.join(np.random.choice(words, size=len(words), replace=True))
                genuine_reviews.append(new_review)
        
        # Create labels
        fake_labels = [1] * len(fake_reviews)
        genuine_labels = [0] * len(genuine_reviews)
        
        # Combine data
        X = fake_reviews + genuine_reviews
        y = fake_labels + genuine_labels
        
        return X, y

    def initialize_model(self, csv_path=None):
        """Initialize and train the ML model"""
        logger.info("Initializing fake review detection model")
        
        # Create a simple pipeline with TF-IDF and Random Forest
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, preprocessor=self.preprocess_text)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Load data from CSV if provided, otherwise use synthetic data
        if csv_path and os.path.exists(csv_path):
            X, y = self.load_dataset_from_csv(csv_path)
        else:
            logger.info("No CSV path provided or file not found, using synthetic data")
            X, y = self._generate_synthetic_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        logger.info(f"Training model on {len(X_train)} samples")
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate model
        score = self.pipeline.score(X_test, y_test)
        logger.info(f"Model accuracy: {score:.4f}")
        
        return score
    
    def analyze_review(self, review_text):
        """Analyze a single review and determine if it's fake"""
        if not review_text:
            return {"error": "Empty review text"}
        
        try:
            # Make prediction
            is_fake_prob = self.pipeline.predict_proba([review_text])[0][1]
            is_fake = is_fake_prob > 0.5
            
            # Extract features for explanation
            features = extract_features(review_text)
            
            # Generate explanation
            explanation = generate_explanation(review_text, is_fake, is_fake_prob, features)
            
            return {
                "is_fake": bool(is_fake),
                "confidence_score": float(is_fake_prob),
                "features": features,
                "explanation": explanation,
                "review": review_text
            }
        except Exception as e:
            logger.error(f"Error analyzing review: {str(e)}")
            return {"error": str(e)}
    
    def analyze_batch(self, reviews):
        """Analyze a batch of reviews"""
        results = []
        for review in reviews:
            if isinstance(review, str):
                results.append(self.analyze_review(review))
            elif isinstance(review, dict) and 'text' in review:
                result = self.analyze_review(review['text'])
                if 'id' in review:
                    result['id'] = review['id']
                results.append(result)
        return results
