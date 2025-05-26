import nltk
import os

# Create NLTK data directory
os.makedirs('/home/runner/nltk_data', exist_ok=True)

# Download necessary NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Open Multilingual WordNet
print("NLTK resources downloaded successfully!")