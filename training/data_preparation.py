import json
import random
from typing import List, Dict
import re

class TrainingDataPreparator:
    """Prepare and augment training data for symptom classification"""
    
    def __init__(self):
        self.symptom_variations = self._load_symptom_variations()
        self.severity_modifiers = self._load_severity_modifiers()
        self.time_expressions = self._load_time_expressions()
    
    def _load_symptom_variations(self) -> Dict[str, List[str]]:
        """Load variations of symptom expressions"""
        return {
            "headache": [
                "headache", "head pain", "head ache", "migraine", "head hurting",
                "head pounding", "head throbbing", "head pressure", "head discomfort"
            ],
            "chest_pain": [
                "chest pain", "chest discomfort", "chest pressure", "chest tightness",
                "heart pain", "heartburn", "chest ache", "chest hurting"
            ],
            "fever": [
                "fever", "high temperature", "running a fever", "hot", "elevated temperature",
                "temperature", "feverish", "burning up"
            ],
            "cough": [
                "cough", "coughing", "dry cough", "wet cough", "productive cough",
                "hacking cough", "persistent cough"
            ],
            "fatigue": [
                "fatigue", "tired", "exhausted", "weak", "lethargic", "no energy",
                "feeling tired", "worn out", "drained"
            ],
            "nausea": [
                "nausea", "sick to stomach", "queasy", "upset stomach", "feeling sick",
                "stomach upset", "nauseous", "sick feeling"
            ]
        }
    
    def _load_severity_modifiers(self) -> Dict[str, List[str]]:
        """Load severity modifiers"""
        return {
            "high": [
                "severe", "intense", "sharp", "sudden", "acute", "terrible",
                "unbearable", "excruciating", "debilitating", "worst", "extreme"
            ],
            "medium": [
                "moderate", "mild", "manageable", "tolerable", "some", "a bit",
                "kind of", "slightly", "fairly"
            ],
            "low": [
                "slight", "minor", "minimal", "barely", "hardly", "just a little",
                "very mild", "slight"
            ]
        }
    
    def _load_time_expressions(self) -> List[str]:
        """Load time expressions"""
        return [
            "recently", "yesterday", "today", "this morning", "this afternoon",
            "for hours", "for days", "for weeks", "suddenly", "gradually",
            "since yesterday", "since this morning", "for the past few days"
        ]
    
    def generate_symptom_variations(self, base_symptom: str, num_variations: int = 5) -> List[str]:
        """Generate variations of a symptom expression"""
        variations = self.symptom_variations.get(base_symptom, [base_symptom])
        
        generated = []
        for _ in range(num_variations):
            # Choose random variation
            symptom = random.choice(variations)
            
            # Add severity modifier
            severity = random.choice(list(self.severity_modifiers.keys()))
            modifier = random.choice(self.severity_modifiers[severity])
            
            # Add time expression
            time_expr = random.choice(self.time_expressions)
            
            # Generate sentence
            templates = [
                f"I have {modifier} {symptom}",
                f"I'm experiencing {modifier} {symptom}",
                f"I feel {modifier} {symptom}",
                f"I have {symptom} that is {modifier}",
                f"My {symptom} is {modifier}",
                f"I have {symptom} {time_expr}",
                f"I'm having {modifier} {symptom} {time_expr}",
                f"I feel {modifier} {symptom} {time_expr}"
            ]
            
            sentence = random.choice(templates)
            generated.append(sentence)
        
        return generated
    
    def augment_training_data(self, original_data: List[Dict], augmentation_factor: int = 3) -> List[Dict]:
        """Augment training data with variations"""
        augmented_data = []
        
        for example in original_data:
            # Add original example
            augmented_data.append(example)
            
            # Generate variations
            variations = self.generate_symptom_variations(
                example['label'], 
                augmentation_factor
            )
            
            for variation in variations:
                augmented_example = {
                    'text': variation,
                    'label': example['label'],
                    'severity': example.get('severity', 'medium'),
                    'keywords': example.get('keywords', [])
                }
                augmented_data.append(augmented_example)
        
        return augmented_data
    
    def create_synthetic_data(self, num_examples_per_category: int = 50) -> List[Dict]:
        """Create synthetic training data"""
        synthetic_data = []
        
        for symptom_category in self.symptom_variations.keys():
            variations = self.generate_symptom_variations(
                symptom_category, 
                num_examples_per_category
            )
            
            for variation in variations:
                # Determine severity based on keywords
                severity = 'medium'
                if any(word in variation.lower() for word in self.severity_modifiers['high']):
                    severity = 'high'
                elif any(word in variation.lower() for word in self.severity_modifiers['low']):
                    severity = 'low'
                
                # Extract keywords
                keywords = []
                for word in variation.lower().split():
                    if word in self.symptom_variations.get(symptom_category, []):
                        keywords.append(word)
                
                synthetic_example = {
                    'text': variation,
                    'label': symptom_category,
                    'severity': severity,
                    'keywords': keywords
                }
                synthetic_data.append(synthetic_example)
        
        return synthetic_data
    
    def prepare_training_file(self, output_path: str, include_synthetic: bool = True):
        """Prepare complete training data file"""
        
        # Load original data
        try:
            with open('app/data/training_data.json', 'r') as f:
                original_data = json.load(f)
        except FileNotFoundError:
            original_data = {'training_examples': []}
        
        # Augment original data
        augmented_data = self.augment_training_data(
            original_data['training_examples'], 
            augmentation_factor=2
        )
        
        # Add synthetic data if requested
        if include_synthetic:
            synthetic_data = self.create_synthetic_data(num_examples_per_category=30)
            augmented_data.extend(synthetic_data)
        
        # Shuffle data
        random.shuffle(augmented_data)
        
        # Split into train/validation
        split_idx = int(0.8 * len(augmented_data))
        train_data = augmented_data[:split_idx]
        val_data = augmented_data[split_idx:]
        
        # Create output structure
        output_data = {
            'training_examples': train_data,
            'validation_examples': val_data,
            'metadata': {
                'total_examples': len(augmented_data),
                'train_examples': len(train_data),
                'validation_examples': len(val_data),
                'categories': list(set(example['label'] for example in augmented_data)),
                'severity_levels': ['low', 'medium', 'high'],
                'description': 'Augmented training data for symptom classification'
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Training data prepared and saved to {output_path}")
        print(f"Total examples: {len(augmented_data)}")
        print(f"Training examples: {len(train_data)}")
        print(f"Validation examples: {len(val_data)}")
        
        return output_data

def main():
    """Main function to prepare training data"""
    preparator = TrainingDataPreparator()
    
    # Prepare training data
    output_path = 'app/data/augmented_training_data.json'
    preparator.prepare_training_file(output_path, include_synthetic=True)
    
    print("\nTraining data preparation completed!")
    print("You can now use this data to train the BioBERT model.")

if __name__ == "__main__":
    main() 