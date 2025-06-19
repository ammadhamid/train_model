#!/usr/bin/env python3
"""
Setup script for AI Symptom Checker
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def download_nltk_data():
    """Download required NLTK data"""
    nltk_script = """
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
print("NLTK data downloaded successfully")
"""
    return run_command(f'python -c "{nltk_script}"', "Downloading NLTK data")

def download_biobert_model():
    """Download BioBERT model"""
    biobert_script = """
from transformers import AutoTokenizer, AutoModel
try:
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    print("BioBERT model downloaded successfully")
except Exception as e:
    print(f"Error downloading BioBERT: {e}")
"""
    return run_command(f'python -c "{biobert_script}"', "Downloading BioBERT model")

def create_directories():
    """Create necessary directories"""
    directories = [
        'models_cache',
        'logs',
        'data/exports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories created")
    return True

def prepare_training_data():
    """Prepare training data"""
    return run_command("python training/data_preparation.py", "Preparing training data")

def run_tests():
    """Run basic tests"""
    return run_command("python -m pytest tests/ -v", "Running tests")

def create_env_file():
    """Create .env file with default configuration"""
    env_content = """# AI Symptom Checker Configuration
SECRET_KEY=your-secret-key-change-in-production
LOG_LEVEL=INFO
MODEL_CACHE_DIR=models_cache
"""
    
    env_file = Path('.env')
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✓ Created .env file")
    else:
        print("✓ .env file already exists")
    
    return True

def main():
    """Main setup function"""
    print("="*50)
    print("AI Symptom Checker Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\nPlease install dependencies manually: pip install -r requirements.txt")
        sys.exit(1)
    
    # Download NLTK data
    if not download_nltk_data():
        print("\nWarning: NLTK data download failed. Some features may not work.")
    
    # Download BioBERT model
    if not download_biobert_model():
        print("\nWarning: BioBERT model download failed. The model will be downloaded on first use.")
    
    # Create environment file
    create_env_file()
    
    # Prepare training data
    if not prepare_training_data():
        print("\nWarning: Training data preparation failed.")
    
    # Run tests
    print("\nRunning basic tests...")
    run_tests()
    
    print("\n" + "="*50)
    print("Setup completed!")
    print("="*50)
    print("\nTo start the API server:")
    print("  python app.py")
    print("\nTo train the model:")
    print("  python training/train_model.py")
    print("\nTo test the API:")
    print("  python tests/test_api.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 