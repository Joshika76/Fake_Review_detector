import re
import numpy as np
from textstat import flesch_reading_ease


def generate_explanation(text, prediction, confidence, feature_importances):
    """
    Generate a human-readable explanation for the prediction.
    
    Args:
        text: Original review text
        prediction: 'genuine' or 'fake'
        confidence: Confidence score (0-1)
        feature_importances: Dict of feature importance values
        
    Returns:
        Dict containing explanation details
    """
    explanation = {
        'prediction': prediction,
        'confidence': round(confidence * 100, 2),
        'summary': '',
        'factors': [],
        'suggestions': []
    }
    
    # Create summary
    if prediction == 'fake':
        explanation['summary'] = "This review has been classified as potentially fake."
    else:
        explanation['summary'] = "This review appears to be genuine."
    
    # Analyze factors
    factors = []
    
    # Check sentiment
    sentiment = feature_importances['sentiment']
    if prediction == 'fake' and abs(sentiment) > 0.75:
        factors.append({
            'name': 'Extreme sentiment',
            'description': f"The review uses {'extremely positive' if sentiment > 0 else 'extremely negative'} language, which is common in fake reviews.",
            'score': abs(sentiment)
        })
    elif prediction == 'genuine' and abs(sentiment) < 0.6:
        factors.append({
            'name': 'Moderate sentiment',
            'description': "The review uses balanced language with moderate sentiment, typical of genuine reviews.",
            'score': 1 - abs(sentiment)
        })
    
    # Check exclamation marks
    excl_count = feature_importances['exclamation'] * 10  # Unnormalize
    if prediction == 'fake' and excl_count > 3:
        factors.append({
            'name': 'Excessive punctuation',
            'description': f"The review contains {int(excl_count)} exclamation marks, which is unusually high.",
            'score': min(excl_count / 5, 1)
        })
    
    # Check capitalization
    cap_ratio = feature_importances['capitalization']
    if prediction == 'fake' and cap_ratio > 0.2:
        factors.append({
            'name': 'Unusual capitalization',
            'description': "The review contains an unusually high amount of capitalized text.",
            'score': min(cap_ratio * 3, 1)
        })
    
    # Check readability
    readability = feature_importances['readability']
    if prediction == 'fake' and readability < 0.4:
        factors.append({
            'name': 'Low readability',
            'description': "The review has poor readability, which can indicate hastily written fake content.",
            'score': 1 - readability
        })
    elif prediction == 'genuine' and readability > 0.6:
        factors.append({
            'name': 'Good readability',
            'description': "The review has good readability, typical of carefully written genuine reviews.",
            'score': readability
        })
    
    # Check personal pronouns
    pronouns = feature_importances['personal_pronouns'] * 10  # Unnormalize
    if prediction == 'fake' and pronouns > 5:
        factors.append({
            'name': 'Excessive personal references',
            'description': "The review contains many personal pronouns, which can indicate fake personal stories.",
            'score': min(pronouns / 7, 1)
        })
    
    # Review length
    length = feature_importances['review_length'] * 100  # Unnormalize
    if prediction == 'fake' and (length < 20 or length > 200):
        factors.append({
            'name': 'Unusual length',
            'description': f"The review is {'very short' if length < 20 else 'excessively long'}, which can be a red flag.",
            'score': 0.7
        })
    
    # Add model contribution information
    model_contribs = feature_importances['model_contributions']
    strongest_model = max(model_contribs.items(), key=lambda x: abs(x[1] - 0.5))
    
    model_names = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'svm': 'Support Vector Machine'
    }
    
    factors.append({
        'name': 'Model consensus',
        'description': f"The {model_names[strongest_model[0]]} model was most confident in this classification.",
        'score': abs(strongest_model[1] - 0.5) * 2  # Scale to 0-1
    })
    
    # Sort factors by score
    explanation['factors'] = sorted(factors, key=lambda x: x['score'], reverse=True)
    
    # Generate suggestions if it's likely a fake review
    if prediction == 'fake' and confidence > 0.6:
        suggestions = []
        
        if excl_count > 3:
            suggestions.append("Reduce the use of exclamation marks")
            
        if cap_ratio > 0.2:
            suggestions.append("Avoid excessive use of capital letters")
            
        if abs(sentiment) > 0.75:
            suggestions.append("Use more balanced language instead of extreme expressions")
            
        if readability < 0.4:
            suggestions.append("Improve the clarity and structure of the text")
            
        if len(suggestions) == 0:
            suggestions.append("Provide more specific details about your experience")
            
        explanation['suggestions'] = suggestions
    
    return explanation
