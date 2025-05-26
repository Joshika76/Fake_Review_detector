# utils.py - Utility functions for the application
import re
from collections import Counter
import numpy as np

def count_exclamation_marks(text):
    """Count the number of exclamation marks in the text"""
    return text.count('!')

def count_uppercase_words(text):
    """Count the number of words in ALL CAPS"""
    words = text.split()
    uppercase_words = [word for word in words if word.isupper() and len(word) > 1]
    return len(uppercase_words)

def count_superlatives(text):
    """Count superlative words like 'best', 'greatest', etc."""
    superlative_words = ['best', 'greatest', 'amazing', 'incredible', 'perfect', 
                         'excellent', 'outstanding', 'exceptional', 'fantastic', 
                         'superb', 'wonderful', 'awesome', 'brilliant', 'spectacular']
    text_lower = text.lower()
    count = 0
    for word in superlative_words:
        count += len(re.findall(r'\b' + word + r'\b', text_lower))
    return count

def count_personal_pronouns(text):
    """Count personal pronouns (I, me, my, etc.)"""
    pronouns = ['i', 'me', 'my', 'mine', 'myself']
    text_lower = text.lower()
    count = 0
    for pronoun in pronouns:
        count += len(re.findall(r'\b' + pronoun + r'\b', text_lower))
    return count

def calculate_word_diversity(text):
    """Calculate lexical diversity (unique words / total words)"""
    words = text.lower().split()
    if not words:
        return 0
    return len(set(words)) / len(words)

def calculate_average_word_length(text):
    """Calculate average word length"""
    words = text.split()
    if not words:
        return 0
    return sum(len(word) for word in words) / len(words)

def count_number_of_sentences(text):
    """Count the number of sentences"""
    # Simple sentence counting based on '.', '!', or '?' followed by a space and uppercase letter
    import re
    sentence_endings = re.findall(r'[.!?][\s]+[A-Z]', text)
    # Add 1 for the last sentence which may not match the pattern
    return len(sentence_endings) + 1

def calculate_average_sentence_length(text):
    """Calculate average sentence length in words"""
    # Simple sentence splitting on common sentence endings
    import re
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0
    
    words_per_sentence = [len(sentence.split()) for sentence in sentences]
    return sum(words_per_sentence) / len(sentences)

def extract_features(text):
    """Extract features from the text for explaining the decision"""
    return {
        "exclamation_marks": count_exclamation_marks(text),
        "uppercase_words": count_uppercase_words(text),
        "superlatives": count_superlatives(text),
        "personal_pronouns": count_personal_pronouns(text),
        "word_diversity": calculate_word_diversity(text),
        "avg_word_length": calculate_average_word_length(text),
        "sentence_count": count_number_of_sentences(text),
        "avg_sentence_length": calculate_average_sentence_length(text),
        "text_length": len(text)
    }

def generate_explanation(text, is_fake, confidence_score, features):
    """Generate an explanation for the classification decision"""
    
    # Initialize explanation
    explanation = []
    
    # Analyze results
    if is_fake:
        explanation.append(f"This review has been classified as likely FAKE with {confidence_score:.1%} confidence.")
    else:
        explanation.append(f"This review has been classified as likely GENUINE with {(1-confidence_score):.1%} confidence.")
    
    # Add feature-specific explanations
    considerations = []
    
    # Exclamation marks
    if features["exclamation_marks"] > 3:
        considerations.append(f"High number of exclamation marks ({features['exclamation_marks']}), which is common in fake reviews")
    
    # Uppercase words
    if features["uppercase_words"] > 2:
        considerations.append(f"Contains {features['uppercase_words']} words in ALL CAPS, often used for emphasis in fake reviews")
    
    # Superlatives
    if features["superlatives"] > 2:
        considerations.append(f"Uses many superlative words ({features['superlatives']}), like 'best', 'amazing', etc.")
    
    # Personal pronouns
    if features["personal_pronouns"] < 2 and len(text) > 100:
        considerations.append("Uses few personal pronouns, which is unusual for authentic reviews")
    
    # Word diversity
    if features["word_diversity"] < 0.6:
        considerations.append("Has low vocabulary diversity, which can indicate generic content")
    elif features["word_diversity"] > 0.8:
        considerations.append("Has high vocabulary diversity, suggesting more thoughtful content")
    
    # Sentence structure
    if features["avg_sentence_length"] > 20:
        considerations.append("Contains unusually long sentences")
    elif features["avg_sentence_length"] < 5 and features["sentence_count"] > 3:
        considerations.append("Contains many short sentences, which can be a sign of exaggeration")
    
    # Length
    if features["text_length"] < 50:
        considerations.append("Very short review, providing little detailed information")
    elif features["text_length"] > 500:
        considerations.append("Unusually long review, which can indicate over-justification")
    
    # Combine all considerations
    if considerations:
        explanation.append("\nKey factors in this analysis:")
        for i, consideration in enumerate(considerations, 1):
            explanation.append(f"{i}. {consideration}")
    
    # General explanation
    explanation.append("\nNote: This analysis is based on linguistic patterns and statistical models. Context matters, and no algorithm is 100% accurate.")
    
    return "\n".join(explanation)
