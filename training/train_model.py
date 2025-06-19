import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    AdamW, get_linear_schedule_with_warmup
)
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import argparse
import os
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SymptomDataset(Dataset):
    """Dataset for symptom classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        self.label_to_id = {label: idx for idx, label in enumerate(set(labels))}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_to_id[label], dtype=torch.long)
        }

class BioBERTSymptomClassifier(nn.Module):
    """BioBERT-based classifier for symptom classification"""
    
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super(BioBERTSymptomClassifier, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return loss, logits

def load_training_data(data_path: str) -> Tuple[List[str], List[str]]:
    """Load training data from JSON file"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    texts = []
    labels = []
    
    for example in data['training_examples']:
        texts.append(example['text'])
        labels.append(example['label'])
    
    return texts, labels

def prepare_data(texts: List[str], labels: List[str], tokenizer, 
                test_size: float = 0.2, max_length: int = 512):
    """Prepare data for training"""
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = SymptomDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = SymptomDataset(val_texts, val_labels, tokenizer, max_length)
    
    return train_dataset, val_dataset

def train_model(model, train_dataloader, val_dataloader, 
               num_epochs: int, learning_rate: float, device):
    """Train the model"""
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # Training loop
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        total_train_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            
            loss, logits = model(input_ids, attention_mask, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                loss, logits = model(input_ids, attention_mask, labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = accuracy_score(true_labels, predictions)
        
        logger.info(f"Average validation loss: {avg_val_loss:.4f}")
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info("Saved best model")
    
    return model

def evaluate_model(model, test_dataloader, device, id_to_label):
    """Evaluate the trained model"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            _, logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Convert IDs back to labels
    pred_labels = [id_to_label[pred] for pred in predictions]
    true_labels = [id_to_label[label] for label in true_labels]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))
    
    return accuracy_score(true_labels, pred_labels)

def main():
    parser = argparse.ArgumentParser(description='Train BioBERT symptom classifier')
    parser.add_argument('--data_path', type=str, default='app/data/training_data.json',
                       help='Path to training data JSON file')
    parser.add_argument('--model_name', type=str, 
                       default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                       help='Pre-trained model name')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='trained_model',
                       help='Directory to save trained model')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load data
    logger.info("Loading training data...")
    texts, labels = load_training_data(args.data_path)
    logger.info(f"Loaded {len(texts)} training examples")
    
    # Prepare data
    train_dataset, val_dataset = prepare_data(texts, labels, tokenizer, max_length=args.max_length)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    num_labels = len(set(labels))
    model = BioBERTSymptomClassifier(args.model_name, num_labels)
    model.to(device)
    
    # Train model
    logger.info("Starting training...")
    trained_model = train_model(model, train_dataloader, val_dataloader, 
                               args.num_epochs, args.learning_rate, device)
    
    # Save model and tokenizer
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    trained_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save label mapping
    label_mapping = {
        'label_to_id': train_dataset.label_to_id,
        'id_to_label': train_dataset.id_to_label
    }
    
    with open(os.path.join(args.output_dir, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f)
    
    logger.info(f"Model saved to {args.output_dir}")
    
    # Evaluate on validation set
    logger.info("Evaluating model...")
    accuracy = evaluate_model(trained_model, val_dataloader, device, train_dataset.id_to_label)
    logger.info(f"Final validation accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 