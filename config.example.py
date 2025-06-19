import os

class Config:
    """Example configuration file. Copy to config.py and fill in real values."""
    SECRET_KEY = 'your-secret-key-here'
    
    # Model configuration
    MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    
    # Performance configuration
    USE_GPU = False
    DEVICE = 'cpu'
    MODEL_CACHE_SIZE = 1000
    RESPONSE_CACHE_TTL = 3600  # 1 hour
    
    # API configuration
    API_TITLE = 'AI Symptom Checker API'
    API_VERSION = '1.0.0'
    
    # Medical data paths
    SYMPTOMS_DB_PATH = 'app/data/symptoms_database.json'
    TRAINING_DATA_PATH = 'app/data/training_data.json'
    
    # Model cache directory
    MODEL_CACHE_DIR = 'app/models_cache'
    
    # Redis configuration for caching
    REDIS_URL = 'redis://localhost:6379'
    REDIS_DB = 0
    
    # Logging
    LOG_LEVEL = 'INFO'
    
    # Security
    ALLOWED_CORS_ORIGINS = ['http://localhost:3000']
    JWT_SECRET = 'your-jwt-secret-here'
    API_KEY = 'your-api-key-here'
    RATE_LIMIT = '10/minute' 