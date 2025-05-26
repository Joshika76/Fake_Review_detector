import numpy as np
from sklearn.ensemble import VotingClassifier


def ensemble_prediction(lr_prob, rf_prob, svm_prob, weights=None):
    """
    Combine predictions from multiple models using weighted averaging.
    
    Args:
        lr_prob: Probability from Logistic Regression
        rf_prob: Probability from Random Forest
        svm_prob: Probability from SVM
        weights: Optional list of weights for each model
        
    Returns:
        final_prob: Final probability after ensemble
        prediction: 'genuine' or 'fake' based on threshold
    """
    # If weights are not provided, use equal weights
    if weights is None:
        weights = [1/3, 1/3, 1/3]
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Combine probabilities
    probs = np.array([lr_prob, rf_prob, svm_prob])
    final_prob = np.dot(weights, probs)
    
    # Make prediction based on threshold
    prediction = 'genuine' if final_prob >= 0.5 else 'fake'
    
    return final_prob, prediction


def dynamic_weighting(lr_accuracy, rf_accuracy, svm_accuracy):
    """
    Dynamically assign weights to models based on their performance.
    
    Args:
        lr_accuracy: Accuracy of Logistic Regression model
        rf_accuracy: Accuracy of Random Forest model
        svm_accuracy: Accuracy of SVM model
        
    Returns:
        List of weights for each model
    """
    # Calculate weights proportionally to accuracy
    total_accuracy = lr_accuracy + rf_accuracy + svm_accuracy
    
    if total_accuracy == 0:
        return [1/3, 1/3, 1/3]
    
    lr_weight = lr_accuracy / total_accuracy
    rf_weight = rf_accuracy / total_accuracy
    svm_weight = svm_accuracy / total_accuracy
    
    return [lr_weight, rf_weight, svm_weight]
