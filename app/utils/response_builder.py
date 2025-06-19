from typing import Dict, List, Optional
import random

class ResponseBuilder:
    """Build natural language responses for the symptom checker"""
    
    def __init__(self):
        self.response_templates = self._load_response_templates()
    
    def _load_response_templates(self) -> Dict:
        """Load response templates for different scenarios"""
        return {
            "greeting": [
                "Hello! I'm here to help you understand your symptoms. What are you experiencing today?",
                "Hi there! I can help you assess your symptoms. What's bothering you?",
                "Welcome! I'm your AI health assistant. What symptoms are you experiencing?"
            ],
            "symptom_analysis": {
                "high_severity": [
                    "I understand you're experiencing {symptom}. This could be serious and requires immediate attention. ",
                    "Your {symptom} sounds concerning. This needs prompt medical evaluation. ",
                    "Based on your description of {symptom}, this appears to be a serious situation. "
                ],
                "medium_severity": [
                    "I see you're dealing with {symptom}. Let me ask you some questions to better understand your situation. ",
                    "You're experiencing {symptom}. I'd like to gather more information to help you. ",
                    "I understand you have {symptom}. Let me get some additional details. "
                ],
                "low_severity": [
                    "I understand you have {symptom}. This is usually manageable. ",
                    "You're experiencing {symptom}. This is typically not serious. ",
                    "I see you have {symptom}. This is generally mild and treatable. "
                ]
            },
            "question_intro": [
                "To better understand your situation, could you tell me: ",
                "To help you properly, I need to know: ",
                "Let me ask you a few questions: ",
                "To provide the best guidance, please tell me: "
            ],
            "follow_up": [
                "Thank you for that information. Now, could you tell me: ",
                "Based on what you've shared, I'd like to know: ",
                "That's helpful. One more question: ",
                "Thanks for clarifying. Let me ask: "
            ],
            "recommendations": {
                "high": [
                    "Given the severity of your symptoms, I strongly recommend seeking immediate medical attention.",
                    "Your symptoms require prompt medical evaluation. Please contact a healthcare provider right away.",
                    "This is a serious situation that needs immediate medical care."
                ],
                "medium": [
                    "I recommend consulting a healthcare provider to properly evaluate your symptoms.",
                    "It would be wise to see a doctor to get a proper assessment.",
                    "Consider scheduling an appointment with your healthcare provider."
                ],
                "low": [
                    "Your symptoms are typically manageable with self-care.",
                    "This is usually not serious and can be treated at home.",
                    "These symptoms generally resolve on their own with proper care."
                ]
            },
            "disclaimer": [
                "Remember, I'm here to provide information only. Always consult with qualified healthcare providers for medical advice.",
                "This information is for educational purposes. Please consult a healthcare professional for proper diagnosis.",
                "I can help guide you, but professional medical evaluation is always recommended."
            ]
        }
    
    def build_greeting_response(self) -> str:
        """Build a greeting response"""
        return random.choice(self.response_templates["greeting"])
    
    def build_symptom_response(self, symptom_category: str, severity: str, 
                              confidence: float, questions: List[str]) -> str:
        """Build a response for symptom analysis"""
        
        # Get severity-specific template
        severity_key = f"{severity}_severity"
        templates = self.response_templates["symptom_analysis"].get(severity_key, 
                                                                   self.response_templates["symptom_analysis"]["medium_severity"])
        
        # Build the response
        response = random.choice(templates).format(symptom=symptom_category.replace('_', ' '))
        
        # Add confidence indicator if low
        if confidence < 0.7:
            response += "I'm not entirely certain about this assessment, so "
        
        # Add question intro
        response += random.choice(self.response_templates["question_intro"])
        
        # Add the first question
        if questions:
            response += questions[0]
        else:
            response += "How long have you been experiencing this?"
        
        return response
    
    def build_chat_response(self, message: str, analysis_result: Dict) -> str:
        """Build a chat response based on analysis"""
        
        symptom_category = analysis_result.get('symptom_category', 'symptoms')
        severity = analysis_result.get('severity', 'medium')
        questions = analysis_result.get('follow_up_questions', [])
        
        # Start with symptom acknowledgment
        response = self.build_symptom_response(symptom_category, severity, 
                                             analysis_result.get('confidence', 0.8), questions)
        
        # Add recommendations if available
        recommendations = analysis_result.get('recommendations', [])
        if recommendations:
            response += "\n\n" + random.choice(self.response_templates["recommendations"][severity])
        
        # Add disclaimer
        response += "\n\n" + random.choice(self.response_templates["disclaimer"])
        
        return response
    
    def build_follow_up_response(self, user_answer: str, next_question: str) -> str:
        """Build a response for follow-up questions"""
        response = random.choice(self.response_templates["follow_up"])
        response += next_question
        return response
    
    def build_recommendation_response(self, recommendations: List[str], severity: str) -> str:
        """Build a response focused on recommendations"""
        response = random.choice(self.response_templates["recommendations"][severity])
        
        if recommendations:
            response += "\n\nSpecific recommendations:\n"
            for i, rec in enumerate(recommendations[:3], 1):  # Limit to 3 recommendations
                response += f"{i}. {rec}\n"
        
        response += "\n" + random.choice(self.response_templates["disclaimer"])
        return response
    
    def build_error_response(self, error_type: str) -> str:
        """Build an error response"""
        error_responses = {
            "invalid_symptom": "I'm sorry, but I couldn't identify any specific symptoms in your message. Could you please describe what you're experiencing in more detail?",
            "unclear": "I'm having trouble understanding your symptoms. Could you please be more specific about what you're experiencing?",
            "too_long": "Your message is quite long. Could you please focus on your main symptoms?",
            "no_medical_terms": "I didn't detect any medical symptoms in your message. Please describe your health concerns."
        }
        
        return error_responses.get(error_type, "I'm sorry, but I couldn't process your message. Please try again.")
    
    def build_summary_response(self, conversation_history: List[Dict]) -> str:
        """Build a summary response based on conversation history"""
        if not conversation_history:
            return "I don't have enough information to provide a summary yet."
        
        # Extract key information
        symptoms = []
        severities = []
        
        for msg in conversation_history:
            if 'symptom_category' in msg:
                symptoms.append(msg['symptom_category'])
            if 'severity' in msg:
                severities.append(msg['severity'])
        
        if not symptoms:
            return "I don't have enough information to provide a summary yet."
        
        # Build summary
        response = "Based on our conversation, you've mentioned: "
        response += ", ".join(set(symptoms)) + ". "
        
        if severities:
            max_severity = max(severities, key=lambda x: ['low', 'medium', 'high'].index(x))
            response += f"The overall severity appears to be {max_severity}. "
        
        response += "\n\n" + random.choice(self.response_templates["disclaimer"])
        
        return response 