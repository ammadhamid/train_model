from typing import Dict, List, Optional
from .biobert_classifier import SymptomClassifier
from .question_generator import QuestionGenerator
import json
import re

class SymptomAnalyzer:
    """Main symptom analysis orchestrator"""
    
    def __init__(self):
        self.classifier = SymptomClassifier()
        self.question_generator = QuestionGenerator()
        self.symptoms_db = self._load_symptoms_database()
    
    def _load_symptoms_database(self) -> Dict:
        """Load the symptoms database"""
        try:
            with open('app/data/symptoms_database.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_symptoms_db()
    
    def _get_default_symptoms_db(self) -> Dict:
        """Default symptoms database if file not found"""
        return {
            "headache": {
                "keywords": ["headache", "head pain", "migraine", "head ache"],
                "questions": [
                    "How long have you had this headache?",
                    "Is the pain on one side or both sides?",
                    "Do you also feel nausea or sensitivity to light?",
                    "What type of pain is it? (throbbing, sharp, dull)",
                    "Have you had similar headaches before?"
                ],
                "severity_indicators": {
                    "high": ["severe", "intense", "sudden", "worse", "unbearable"],
                    "medium": ["moderate", "mild", "manageable"],
                    "low": ["slight", "minor", "tolerable"]
                },
                "recommendations": {
                    "low": ["Rest in a quiet, dark room", "Consider over-the-counter pain relievers"],
                    "medium": ["Monitor symptoms", "Consider consulting a doctor if persistent"],
                    "high": ["Seek immediate medical attention", "Consider emergency care"]
                }
            },
            "chest_pain": {
                "keywords": ["chest pain", "chest discomfort", "heart pain", "chest tightness"],
                "questions": [
                    "Do you smoke or have a history of heart disease?",
                    "Is the pain sharp or dull?",
                    "Does the pain radiate to your arm, jaw, or back?",
                    "What were you doing when the pain started?",
                    "Do you have shortness of breath or sweating?"
                ],
                "severity_indicators": {
                    "high": ["severe", "crushing", "sudden", "radiating", "sweating"],
                    "medium": ["moderate", "pressure", "tightness"],
                    "low": ["mild", "slight", "intermittent"]
                },
                "recommendations": {
                    "low": ["Monitor symptoms", "Consider lifestyle changes"],
                    "medium": ["Consult a doctor", "Monitor for worsening symptoms"],
                    "high": ["Seek emergency medical attention immediately", "Call emergency services"]
                }
            },
            "fever": {
                "keywords": ["fever", "high temperature", "hot", "chills"],
                "questions": [
                    "What is your temperature?",
                    "How long have you had the fever?",
                    "Do you have other symptoms like cough or sore throat?",
                    "Have you been exposed to sick people recently?",
                    "Do you have any chronic medical conditions?"
                ],
                "severity_indicators": {
                    "high": ["very high", "above 103", "severe", "persistent"],
                    "medium": ["moderate", "around 101-102", "intermittent"],
                    "low": ["low grade", "mild", "below 100"]
                },
                "recommendations": {
                    "low": ["Rest and stay hydrated", "Monitor temperature"],
                    "medium": ["Consider fever-reducing medication", "Consult doctor if persistent"],
                    "high": ["Seek medical attention", "Consider emergency care"]
                }
            }
        }
    
    def analyze_symptom(self, symptom_text: str, session_context: Optional[Dict] = None) -> Dict:
        """Analyze a symptom and return comprehensive analysis"""
        
        # Classify the symptom
        classification = self.classifier.classify_symptom(symptom_text)
        
        # Get symptom category
        category = classification['symptom_category']
        
        # Get symptom data from database
        symptom_data = self.symptoms_db.get(category, {})
        
        # Generate follow-up questions
        questions = self.question_generator.generate_questions(
            symptom_text, 
            category, 
            symptom_data,
            session_context
        )
        
        # Get recommendations based on severity
        recommendations = self._get_recommendations(category, classification['severity'])
        
        # Build response
        response = {
            'symptom_category': category,
            'confidence': classification['confidence'],
            'severity': classification['severity'],
            'follow_up_questions': questions,
            'recommendations': recommendations,
            'analysis': {
                'detected_keywords': self._extract_keywords(symptom_text, symptom_data),
                'symptom_description': self._generate_description(category, classification['severity'])
            }
        }
        
        return response
    
    def _extract_keywords(self, text: str, symptom_data: Dict) -> List[str]:
        """Extract relevant keywords from symptom text"""
        keywords = symptom_data.get('keywords', [])
        detected = []
        
        for keyword in keywords:
            if keyword.lower() in text.lower():
                detected.append(keyword)
        
        return detected
    
    def _generate_description(self, category: str, severity: str) -> str:
        """Generate a description of the symptom"""
        descriptions = {
            'headache': {
                'low': 'Mild headache that may be manageable with rest',
                'medium': 'Moderate headache that may require attention',
                'high': 'Severe headache that needs medical evaluation'
            },
            'chest_pain': {
                'low': 'Mild chest discomfort that should be monitored',
                'medium': 'Moderate chest pain that requires medical consultation',
                'high': 'Severe chest pain that needs immediate medical attention'
            },
            'fever': {
                'low': 'Low-grade fever that may resolve with rest',
                'medium': 'Moderate fever that should be monitored',
                'high': 'High fever that requires medical attention'
            }
        }
        
        return descriptions.get(category, {}).get(severity, f'{severity} {category}')
    
    def _get_recommendations(self, category: str, severity: str) -> List[str]:
        """Get recommendations based on symptom category and severity"""
        symptom_data = self.symptoms_db.get(category, {})
        recommendations = symptom_data.get('recommendations', {})
        
        return recommendations.get(severity, [
            "Monitor your symptoms",
            "Consider consulting a healthcare provider if symptoms persist"
        ])
    
    def process_conversation(self, message: str, conversation_history: List[Dict]) -> Dict:
        """Process a message in the context of a conversation"""
        
        # Analyze current symptom
        analysis = self.analyze_symptom(message, {
            'conversation_history': conversation_history
        })
        
        # Add conversation context
        analysis['conversation_context'] = {
            'message_count': len(conversation_history) + 1,
            'previous_symptoms': [msg.get('symptom_category') for msg in conversation_history[-3:]]
        }
        
        return analysis