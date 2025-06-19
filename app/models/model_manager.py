import torch
import hashlib
import json
import asyncio
from typing import Dict, Optional, Any, List
from functools import lru_cache
import redis
from app.models.biobert_classifier import SymptomClassifier
from app.models.symptom_analyzer import SymptomAnalyzer
from config import Config
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Singleton model manager for efficient model loading and caching"""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = False
            self._classifier = None
            self._analyzer = None
            self._redis = None
            self._device = Config.DEVICE
            self._model_cache = {}
            self._response_cache = {}
    
    async def initialize(self):
        """Initialize models and Redis connection"""
        if self.initialized:
            return
        
        async with self._lock:
            if self.initialized:
                return
            
            logger.info(f"Initializing ModelManager on device: {self._device}")
            
            # Initialize Redis for caching
            try:
                self._redis = redis.from_url(
                    Config.REDIS_URL,
                    db=Config.REDIS_DB,
                    encoding="utf-8",
                    decode_responses=True
                )
                self._redis.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory cache: {e}")
                self._redis = None
            
            # Load models
            try:
                self._classifier = SymptomClassifier()
                self._analyzer = SymptomAnalyzer()
                
                # Move models to appropriate device
                if hasattr(self._classifier, 'model') and self._classifier.model:
                    self._classifier.model.to(self._device)
                
                logger.info(f"Models loaded successfully on {self._device}")
            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                raise
            
            self.initialized = True
    
    @property
    def classifier(self) -> SymptomClassifier:
        """Get the symptom classifier"""
        if not self.initialized:
            raise RuntimeError("ModelManager not initialized. Call initialize() first.")
        return self._classifier
    
    @property
    def analyzer(self) -> SymptomAnalyzer:
        """Get the symptom analyzer"""
        if not self.initialized:
            raise RuntimeError("ModelManager not initialized. Call initialize() first.")
        return self._analyzer
    
    def _generate_cache_key(self, text: str, operation: str) -> str:
        """Generate a cache key for the given text and operation"""
        content = f"{operation}:{text.lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response from Redis or memory"""
        if self._redis:
            try:
                cached = self._redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Fallback to in-memory cache
        return self._response_cache.get(cache_key)
    
    def set_cached_response(self, cache_key: str, response: Dict):
        """Cache response in Redis or memory"""
        if self._redis:
            try:
                self._redis.setex(
                    cache_key,
                    Config.RESPONSE_CACHE_TTL,
                    json.dumps(response)
                )
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Fallback to in-memory cache
        self._response_cache[cache_key] = response
        
        # Limit in-memory cache size
        if len(self._response_cache) > Config.MODEL_CACHE_SIZE:
            # Remove oldest entries
            oldest_keys = list(self._response_cache.keys())[:100]
            for key in oldest_keys:
                del self._response_cache[key]
    
    async def analyze_symptom_cached(self, symptom_text: str, session_context: Optional[Dict] = None) -> Dict:
        """Analyze symptom with caching"""
        cache_key = self._generate_cache_key(symptom_text, "analyze")
        
        # Check cache first
        cached_result = await self.get_cached_response(cache_key)
        if cached_result:
            logger.info(f"Cache hit for symptom analysis: {symptom_text[:50]}...")
            return cached_result
        
        # Perform analysis (do NOT await, it's sync)
        result = self.analyzer.analyze_symptom(symptom_text, session_context)
        
        # Cache the result
        await self.set_cached_response(cache_key, result)
        
        return result
    
    async def chat_cached(self, message: str, conversation_history: List[Dict]) -> Dict:
        """Process chat with caching"""
        # For chat, we include conversation context in cache key
        context_hash = hashlib.md5(
            json.dumps(conversation_history, sort_keys=True).encode()
        ).hexdigest()[:8]
        cache_key = self._generate_cache_key(f"{message}:{context_hash}", "chat")
        
        # Check cache first
        cached_result = self.get_cached_response(cache_key)
        if cached_result:
            logger.info(f"Cache hit for chat: {message[:50]}...")
            return cached_result
        
        # Process chat
        result = self.analyzer.process_conversation(message, conversation_history)
        
        # Cache the result
        self.set_cached_response(cache_key, result)
        
        return result
    
    async def batch_analyze_symptoms(self, symptoms: List[str]) -> List[Dict]:
        """Batch analyze multiple symptoms for better performance"""
        if not symptoms:
            return []
        
        # Check cache for all symptoms first
        results = []
        uncached_symptoms = []
        uncached_indices = []
        
        for i, symptom in enumerate(symptoms):
            cache_key = self._generate_cache_key(symptom, "analyze")
            cached_result = await self.get_cached_response(cache_key)
            if cached_result:
                results.append(cached_result)
            else:
                uncached_symptoms.append(symptom)
                uncached_indices.append(i)
                results.append(None)  # Placeholder
        
        # Process uncached symptoms in batch if possible
        if uncached_symptoms:
            if len(uncached_symptoms) > 1:
                # Use batch prediction for better performance
                batch_predictions = self.classifier.model.predict_symptoms_batch(uncached_symptoms)
                
                for i, prediction in enumerate(batch_predictions):
                    # Create full analysis result (do NOT await, it's sync)
                    result = self.analyzer.analyze_symptom(uncached_symptoms[i])
                    results[uncached_indices[i]] = result
                    
                    # Cache the result
                    cache_key = self._generate_cache_key(uncached_symptoms[i], "analyze")
                    await self.set_cached_response(cache_key, result)
            else:
                # Process individually for single items
                for i, symptom in enumerate(uncached_symptoms):
                    result = await self.analyze_symptom_cached(symptom)
                    results[uncached_indices[i]] = result
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        if self._redis:
            self._redis.close()
        
        # Clear caches
        self._response_cache.clear()
        self._model_cache.clear()
        
        logger.info("ModelManager cleanup completed")

# Global model manager instance
model_manager = ModelManager()

async def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    await model_manager.initialize()
    return model_manager 