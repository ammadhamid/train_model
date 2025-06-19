from typing import List, Dict, Optional
import random
import re

class QuestionGenerator:
    """Generates context-specific follow-up questions for symptoms"""
    
    def __init__(self):
        self.question_templates = self._load_question_templates()
        self.context_rules = self._load_context_rules()
    
    def _load_question_templates(self) -> Dict:
        """Load question templates for different symptom types"""
        return {
            "headache": {
                "duration": [
                    "How long have you had this headache?",
                    "When did the headache start?",
                    "Is this a new headache or recurring?"
                ],
                "location": [
                    "Is the pain on one side or both sides?",
                    "Where exactly is the pain located?",
                    "Does the pain move around or stay in one place?"
                ],
                "intensity": [
                    "How would you rate the pain on a scale of 1-10?",
                    "Is the pain mild, moderate, or severe?",
                    "What type of pain is it? (throbbing, sharp, dull, pressure)"
                ],
                "triggers": [
                    "What were you doing when the headache started?",
                    "Have you been under stress recently?",
                    "Did you consume any alcohol or caffeine?"
                ],
                "associated_symptoms": [
                    "Do you also feel nausea or sensitivity to light?",
                    "Are you experiencing any vision changes?",
                    "Do you have any neck stiffness?"
                ]
            },
            "chest_pain": {
                "characteristics": [
                    "Is the pain sharp or dull?",
                    "Does it feel like pressure, burning, or stabbing?",
                    "Is the pain constant or does it come and go?"
                ],
                "location": [
                    "Where exactly is the pain located?",
                    "Does the pain radiate to your arm, jaw, or back?",
                    "Is it on the left side, right side, or center?"
                ],
                "triggers": [
                    "What were you doing when the pain started?",
                    "Does the pain get worse with movement or breathing?",
                    "Is it related to physical activity or stress?"
                ],
                "risk_factors": [
                    "Do you smoke or have a history of heart disease?",
                    "Do you have diabetes, high blood pressure, or high cholesterol?",
                    "Is there a family history of heart problems?"
                ],
                "associated_symptoms": [
                    "Do you have shortness of breath or sweating?",
                    "Are you feeling lightheaded or dizzy?",
                    "Do you have any nausea or indigestion?"
                ]
            },
            "fever": {
                "temperature": [
                    "What is your temperature?",
                    "Have you measured your temperature?",
                    "How high is the fever?"
                ],
                "duration": [
                    "How long have you had the fever?",
                    "When did the fever start?",
                    "Is this a new fever or recurring?"
                ],
                "associated_symptoms": [
                    "Do you have other symptoms like cough or sore throat?",
                    "Are you experiencing chills or sweating?",
                    "Do you have any body aches or fatigue?"
                ],
                "exposure": [
                    "Have you been exposed to sick people recently?",
                    "Have you traveled recently?",
                    "Are you in close contact with anyone who is ill?"
                ],
                "medical_history": [
                    "Do you have any chronic medical conditions?",
                    "Are you taking any medications?",
                    "Have you had any recent surgeries or procedures?"
                ]
            }
        }
    
    def _load_context_rules(self) -> Dict:
        """Load rules for context-aware question generation"""
        return {
            "severity_based": {
                "high": {
                    "priority_questions": ["duration", "associated_symptoms"],
                    "max_questions": 3,
                    "urgency_indicators": ["immediate", "severe", "sudden"]
                },
                "medium": {
                    "priority_questions": ["duration", "characteristics", "triggers"],
                    "max_questions": 4,
                    "urgency_indicators": ["moderate", "persistent"]
                },
                "low": {
                    "priority_questions": ["duration", "triggers"],
                    "max_questions": 5,
                    "urgency_indicators": ["mild", "manageable"]
                }
            },
            "conversation_based": {
                "first_message": ["duration", "characteristics"],
                "follow_up": ["associated_symptoms", "triggers"],
                "detailed": ["risk_factors", "medical_history"]
            }
        }
    
    def generate_questions(self, symptom_text: str, category: str, 
                          symptom_data: Dict, session_context: Optional[Dict] = None) -> List[str]:
        """Generate context-specific follow-up questions"""
        
        # Get templates for this symptom category
        templates = self.question_templates.get(category, {})
        
        # Determine severity from symptom text
        severity = self._assess_severity_from_text(symptom_text)
        
        # Get context rules
        context_rules = self.context_rules["severity_based"].get(severity, {})
        
        # Determine question categories to prioritize
        priority_categories = context_rules.get("priority_questions", list(templates.keys()))
        max_questions = context_rules.get("max_questions", 4)
        
        # Consider conversation context
        if session_context and session_context.get('conversation_history'):
            conversation_length = len(session_context['conversation_history'])
            if conversation_length == 0:
                # First message - focus on basic info
                priority_categories = self.context_rules["conversation_based"]["first_message"]
            elif conversation_length < 3:
                # Follow-up - get more details
                priority_categories = self.context_rules["conversation_based"]["follow_up"]
            else:
                # Detailed conversation - comprehensive questions
                priority_categories = self.context_rules["conversation_based"]["detailed"]
        
        # Generate questions
        questions = []
        used_categories = set()
        
        # Add priority questions first
        for category_type in priority_categories:
            if category_type in templates and len(questions) < max_questions:
                category_questions = templates[category_type]
                if category_questions:
                    # Select a question from this category
                    selected_question = self._select_best_question(
                        category_questions, symptom_text, used_categories
                    )
                    if selected_question:
                        questions.append(selected_question)
                        used_categories.add(category_type)
        
        # Fill remaining slots with other categories
        remaining_categories = [cat for cat in templates.keys() if cat not in used_categories]
        random.shuffle(remaining_categories)
        
        for category_type in remaining_categories:
            if len(questions) >= max_questions:
                break
            
            category_questions = templates[category_type]
            if category_questions:
                selected_question = self._select_best_question(
                    category_questions, symptom_text, used_categories
                )
                if selected_question:
                    questions.append(selected_question)
        
        # Ensure we have at least some questions
        if not questions:
            questions = self._get_fallback_questions(category)
        
        return questions[:max_questions]
    
    def _assess_severity_from_text(self, text: str) -> str:
        """Assess severity based on keywords in the text"""
        text_lower = text.lower()
        
        high_severity_words = [
            'severe', 'intense', 'sharp', 'sudden', 'worse', 'unbearable',
            'emergency', 'urgent', 'critical', 'extreme', 'terrible'
        ]
        
        medium_severity_words = [
            'moderate', 'mild', 'slight', 'manageable', 'tolerable',
            'some', 'a bit', 'kind of'
        ]
        
        if any(word in text_lower for word in high_severity_words):
            return 'high'
        elif any(word in text_lower for word in medium_severity_words):
            return 'medium'
        else:
            return 'low'
    
    def _select_best_question(self, questions: List[str], symptom_text: str, 
                            used_categories: set) -> Optional[str]:
        """Select the best question based on context and variety"""
        
        # Avoid duplicate questions
        available_questions = [q for q in questions if q not in used_categories]
        
        if not available_questions:
            return None
        
        # Simple selection - could be enhanced with more sophisticated logic
        return random.choice(available_questions)
    
    def _get_fallback_questions(self, category: str) -> List[str]:
        """Get fallback questions if no specific templates are available"""
        fallback_questions = {
            "headache": [
                "How long have you had this headache?",
                "Is the pain mild, moderate, or severe?",
                "Do you have any other symptoms?"
            ],
            "chest_pain": [
                "How long have you had this chest pain?",
                "Is the pain sharp or dull?",
                "Do you have any other symptoms?"
            ],
            "fever": [
                "What is your temperature?",
                "How long have you had the fever?",
                "Do you have any other symptoms?"
            ]
        }
        
        return fallback_questions.get(category, [
            "How long have you had this symptom?",
            "How severe is it?",
            "Do you have any other symptoms?"
        ])
    
    def generate_follow_up_question(self, previous_question: str, 
                                  user_response: str, category: str) -> str:
        """Generate a follow-up question based on user's response"""
        
        # This could be enhanced with more sophisticated logic
        # For now, return a generic follow-up
        return f"Can you tell me more about that? Specifically, {previous_question.lower()}" 