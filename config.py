import os
from dotenv import load_dotenv
import torch

load_dotenv()

class Config:
    """Application configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Model configuration
    MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    
    # Performance configuration
    USE_GPU = os.environ.get('USE_GPU', 'false').lower() == 'true'
    DEVICE = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
    MODEL_CACHE_SIZE = int(os.environ.get('MODEL_CACHE_SIZE', '1000'))
    RESPONSE_CACHE_TTL = int(os.environ.get('RESPONSE_CACHE_TTL', '3600'))  # 1 hour
    
    # API configuration
    API_TITLE = 'AI Symptom Checker API'
    API_VERSION = '1.0.0'
    
    # Medical data paths
    SYMPTOMS_DB_PATH = 'app/data/symptoms_database.json'
    TRAINING_DATA_PATH = 'app/data/training_data.json'
    
    # Model cache directory
    MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'models_cache')
    
    # Redis configuration for caching
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    REDIS_DB = int(os.environ.get('REDIS_DB', '0'))
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Security
    ALLOWED_CORS_ORIGINS = os.environ.get('ALLOWED_CORS_ORIGINS', 'http://localhost:3000').split(',')
    JWT_SECRET = os.environ.get('JWT_SECRET', 'change-this-jwt-secret')
    API_KEY = os.environ.get('API_KEY', 'change-this-api-key')
    RATE_LIMIT = os.environ.get('RATE_LIMIT', '10/minute') 