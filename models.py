# models.py - Not used in this application as we're not using a database
# This file is included to follow the flask blueprint structure
from dataclasses import dataclass

@dataclass
class AnalysisResult:
    """
    Data class to represent the result of a review analysis
    """
    is_fake: bool
    confidence_score: float
    features: dict
    explanation: str
