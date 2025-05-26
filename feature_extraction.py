import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from textstat import flesch_reading_ease


def extract_features(text, sentiment_analyzer):
    """
    Extract linguistic and statistical features from text.
    
    Args:
        text: String, the raw review text
        sentiment_analyzer: NLTK's SentimentIntensityAnalyzer
        
    Returns:
        Numpy array of features
    """
    # Sentiment analysis
    sentiment_score = sentiment_analyzer.polarity_scores(text)['compound']
    
    # Count exclamation marks
    exclamation_count = text.count('!')
    
    # Count question marks
    question_count = text.count('?')
    
    # Ratio of uppercase letters
    uppercase_count = sum(1 for c in text if c.isupper())
    total_chars = len(text.replace(" ", ""))
    uppercase_ratio = uppercase_count / total_chars if total_chars > 0 else 0
    
    # Average word length
    words = text.split()
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Review length (word count)
    review_length = len(words)
    
    # Count personal pronouns (I, me, my, etc.)
    personal_pronouns = ['i', 'me', 'my', 'mine', 'myself']
    tokens = [word.lower() for word in word_tokenize(text)]
    personal_pronoun_count = sum(token in personal_pronouns for token in tokens)
    
    # Readability score
    readability = flesch_reading_ease(text) / 100  # Normalize to 0-1 range
    
    # Return features as a numpy array
    return np.array([
        sentiment_score,
        min(exclamation_count / 10, 1),  # Cap at 1 for normalization
        min(question_count / 5, 1),  # Cap at 1 for normalization
        uppercase_ratio,
        min(avg_word_length / 10, 1),  # Normalize
        min(review_length / 100, 1),  # Normalize
        min(personal_pronoun_count / 10, 1),  # Normalize
        readability
    ])


def get_important_words(text, vectorizer, model):
    """
    Extract words that contribute most to the classification.
    
    Args:
        text: String, the preprocessed text
        vectorizer: TfidfVectorizer used in the model
        model: Trained model with coefficients
        
    Returns:
        List of (word, importance) tuples
    """
    # Get the feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    # Transform the text using the vectorizer
    text_vector = vectorizer.transform([text])
    
    # Get the indices of non-zero elements in the vector
    non_zero_indices = text_vector.nonzero()[1]
    
    # Get the words that are present in the text
    present_words = [(feature_names[idx], text_vector[0, idx]) for idx in non_zero_indices]
    
    # Sort by TF-IDF score (importance)
    present_words.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top words
    return present_words[:10]
