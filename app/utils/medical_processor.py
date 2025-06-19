import re
import nltk
from typing import List, Dict, Set
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class MedicalTextProcessor:
    """Process and clean medical text for analysis"""
    
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.medical_terms = self._load_medical_terms()
        
    def _load_medical_terms(self) -> Set[str]:
        """Load medical terminology"""
        medical_terms = {
            # Symptoms
            'headache', 'migraine', 'pain', 'ache', 'discomfort', 'soreness',
            'fever', 'temperature', 'hot', 'chills', 'sweating',
            'cough', 'coughing', 'sneeze', 'sneezing',
            'fatigue', 'tired', 'exhausted', 'weak', 'lethargic',
            'nausea', 'queasy', 'sick', 'vomiting', 'upset stomach',
            'chest pain', 'chest discomfort', 'heartburn', 'indigestion',
            'shortness of breath', 'difficulty breathing', 'wheezing',
            'dizziness', 'lightheaded', 'faint', 'vertigo',
            'abdominal pain', 'stomach pain', 'belly ache',
            'back pain', 'joint pain', 'muscle pain',
            'skin rash', 'itching', 'hives', 'redness',
            'vision problems', 'blurred vision', 'eye pain',
            'hearing problems', 'ear pain', 'ringing in ears',
            
            # Severity indicators
            'severe', 'intense', 'sharp', 'sudden', 'acute',
            'moderate', 'mild', 'slight', 'minor',
            'chronic', 'persistent', 'recurring', 'intermittent',
            'worse', 'better', 'improving', 'deteriorating',
            
            # Time indicators
            'recently', 'yesterday', 'today', 'this morning',
            'for days', 'for weeks', 'suddenly', 'gradually',
            
            # Body parts
            'head', 'chest', 'stomach', 'back', 'neck',
            'arms', 'legs', 'hands', 'feet', 'eyes', 'ears'
        }
        return medical_terms
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize medical text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_symptoms(self, text: str) -> List[str]:
        """Extract potential symptoms from text"""
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        
        symptoms = []
        
        # Look for symptom patterns
        for i, word in enumerate(words):
            # Single word symptoms
            if word in self.medical_terms:
                symptoms.append(word)
            
            # Multi-word symptoms
            if i < len(words) - 1:
                bigram = f"{word} {words[i+1]}"
                if bigram in self.medical_terms:
                    symptoms.append(bigram)
            
            if i < len(words) - 2:
                trigram = f"{word} {words[i+1]} {words[i+2]}"
                if trigram in self.medical_terms:
                    symptoms.append(trigram)
        
        return list(set(symptoms))
    
    def extract_severity(self, text: str) -> str:
        """Extract severity indicators from text"""
        text_lower = text.lower()
        
        high_severity_words = [
            'severe', 'intense', 'sharp', 'sudden', 'acute', 'terrible',
            'unbearable', 'excruciating', 'debilitating', 'worst'
        ]
        
        medium_severity_words = [
            'moderate', 'mild', 'manageable', 'tolerable', 'some',
            'a bit', 'kind of', 'slightly'
        ]
        
        low_severity_words = [
            'slight', 'minor', 'minimal', 'barely', 'hardly'
        ]
        
        if any(word in text_lower for word in high_severity_words):
            return 'high'
        elif any(word in text_lower for word in medium_severity_words):
            return 'medium'
        elif any(word in text_lower for word in low_severity_words):
            return 'low'
        else:
            return 'unknown'
    
    def extract_duration(self, text: str) -> str:
        """Extract duration indicators from text"""
        text_lower = text.lower()
        
        duration_patterns = {
            'acute': r'(just|recently|today|this morning|suddenly)',
            'short_term': r'(for hours|for a day|for days)',
            'long_term': r'(for weeks|for months|chronic|persistent)'
        }
        
        for duration_type, pattern in duration_patterns.items():
            if re.search(pattern, text_lower):
                return duration_type
        
        return 'unknown'
    
    def extract_body_location(self, text: str) -> List[str]:
        """Extract body location mentions"""
        body_parts = {
            'head': ['head', 'forehead', 'temple', 'skull'],
            'chest': ['chest', 'breast', 'ribs', 'heart'],
            'abdomen': ['stomach', 'belly', 'abdomen', 'gut'],
            'back': ['back', 'spine', 'shoulder'],
            'limbs': ['arm', 'leg', 'hand', 'foot', 'finger', 'toe'],
            'face': ['face', 'eye', 'ear', 'nose', 'mouth', 'throat']
        }
        
        text_lower = text.lower()
        locations = []
        
        for location, terms in body_parts.items():
            if any(term in text_lower for term in terms):
                locations.append(location)
        
        return locations
    
    def preprocess_for_model(self, text: str) -> Dict:
        """Preprocess text for model input"""
        cleaned_text = self.clean_text(text)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'symptoms': self.extract_symptoms(cleaned_text),
            'severity': self.extract_severity(text),
            'duration': self.extract_duration(text),
            'body_locations': self.extract_body_location(text),
            'word_count': len(cleaned_text.split()),
            'has_medical_terms': bool(self.extract_symptoms(cleaned_text))
        }
    
    def validate_symptom_text(self, text: str) -> Dict:
        """Validate if text contains valid symptom information"""
        if not text or len(text.strip()) < 3:
            return {
                'is_valid': False,
                'error': 'Text too short or empty'
            }
        
        if len(text) > 500:
            return {
                'is_valid': False,
                'error': 'Text too long (max 500 characters)'
            }
        
        processed = self.preprocess_for_model(text)
        
        if not processed['has_medical_terms']:
            return {
                'is_valid': False,
                'error': 'No medical symptoms detected'
            }
        
        return {
            'is_valid': True,
            'processed_data': processed
        } 