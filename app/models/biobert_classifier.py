import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os

class BioBERTClassifier(nn.Module):
    """BioBERT-based classifier for medical symptom classification"""
    
    def __init__(self, model_name: str, num_labels: int, max_length: int = 512):
        super(BioBERTClassifier, self).__init__()
        self.max_length = max_length
        
        # Load BioBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Add classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Load symptom categories
        self.symptom_categories = self._load_symptom_categories()
        
        # Move to appropriate device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def _load_symptom_categories(self) -> List[str]:
        """Load symptom categories from database"""
        try:
            with open('app/data/symptoms_database.json', 'r') as f:
                data = json.load(f)
                return list(data.keys())
        except FileNotFoundError:
            # Default categories if file not found
            return [
                'headache', 'chest_pain', 'fever', 'cough', 'fatigue',
                'nausea', 'dizziness', 'shortness_of_breath', 'abdominal_pain',
                'back_pain', 'joint_pain', 'skin_rash', 'vision_problems',
                'hearing_problems', 'sleep_problems', 'anxiety', 'depression'
            ]
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass through the model"""
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def predict_symptom(self, text: str) -> Dict:
        """Predict symptom category from text"""
        self.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'symptom_category': self.symptom_categories[predicted_class],
            'confidence': confidence,
            'all_probabilities': probabilities[0].cpu().tolist(),
            'categories': self.symptom_categories
        }
    
    def predict_symptoms_batch(self, texts: List[str]) -> List[Dict]:
        """Predict symptom categories for multiple texts (batch processing)"""
        self.eval()
        
        if not texts:
            return []
        
        # Tokenize all inputs
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            confidences = torch.gather(probabilities, 1, predicted_classes.unsqueeze(1)).squeeze(1)
        
        results = []
        for i in range(len(texts)):
            results.append({
                'symptom_category': self.symptom_categories[predicted_classes[i].item()],
                'confidence': confidences[i].item(),
                'all_probabilities': probabilities[i].cpu().tolist(),
                'categories': self.symptom_categories
            })
        
        return results
    
    def get_symptom_embeddings(self, text: str) -> np.ndarray:
        """Get BioBERT embeddings for a symptom text"""
        self.eval()
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings

class SymptomClassifier:
    """High-level interface for symptom classification"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the BioBERT classifier model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load fine-tuned model
                self.model = BioBERTClassifier.from_pretrained(self.model_path)
            else:
                # Load base model
                self.model = BioBERTClassifier(
                    model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                    num_labels=17  # Number of symptom categories
                )
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to base model
            self.model = BioBERTClassifier(
                model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                num_labels=17
            )
    
    def classify_symptom(self, symptom_text: str) -> Dict:
        """Classify a symptom and return detailed analysis"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Basic text preprocessing
        symptom_text = symptom_text.lower().strip()
        
        # Get prediction
        prediction = self.model.predict_symptom(symptom_text)
        
        # Add severity assessment
        severity = self._assess_severity(symptom_text, prediction['symptom_category'])
        prediction['severity'] = severity
        
        return prediction
    
    def _assess_severity(self, text: str, category: str) -> str:
        """Assess symptom severity based on keywords and category"""
        # High severity keywords
        high_severity_keywords = [
            'severe', 'intense', 'sharp', 'sudden', 'worse', 'unbearable',
            'emergency', 'urgent', 'critical', 'extreme'
        ]
        
        # Medium severity keywords
        medium_severity_keywords = [
            'moderate', 'mild', 'slight', 'manageable', 'tolerable'
        ]
        
        # Check for severity indicators
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in high_severity_keywords):
            return 'high'
        elif any(keyword in text_lower for keyword in medium_severity_keywords):
            return 'medium'
        else:
            return 'low'
    
    def get_similar_symptoms(self, symptom_text: str, top_k: int = 5) -> List[Dict]:
        """Find similar symptoms based on embeddings"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Get embedding for input symptom
        input_embedding = self.model.get_symptom_embeddings(symptom_text)
        
        # This would typically compare against a database of symptoms
        # For now, return empty list
        return [] 