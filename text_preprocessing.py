import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get stopwords
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Preprocess the text for machine learning.
    
    Args:
        text: String, the raw review text
        
    Returns:
        String, the preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text


def get_text_stats(text):
    """
    Extract basic statistics from the text.
    
    Args:
        text: String, the raw review text
        
    Returns:
        Dict with text statistics
    """
    # Count words
    word_count = len(text.split())
    
    # Count sentences
    sentence_count = len(re.split(r'[.!?]+', text))
    
    # Count exclamation marks
    exclamation_count = text.count('!')
    
    # Count question marks
    question_count = text.count('?')
    
    # Count uppercase letters
    uppercase_count = sum(1 for c in text if c.isupper())
    
    # Ratio of uppercase to all characters
    total_chars = len(text.replace(" ", ""))
    uppercase_ratio = uppercase_count / total_chars if total_chars > 0 else 0
    
    # Calculate average word length
    words = text.split()
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'uppercase_ratio': uppercase_ratio,
        'avg_word_length': avg_word_length
    }
