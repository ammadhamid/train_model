from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import jwt
import bleach
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.models.model_manager import get_model_manager
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create router
router = APIRouter()

# Pydantic models
class SymptomRequest(BaseModel):
    symptom: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    session_id: str
    conversation_history: Optional[List[Dict[str, Any]]] = []

class SymptomResponse(BaseModel):
    symptom_category: str
    confidence: float
    severity: str
    follow_up_questions: List[str]
    recommendations: List[str]
    analysis: Dict[str, Any]

class ChatResponse(BaseModel):
    response: str
    symptom_category: Optional[str] = None
    confidence: Optional[float] = None
    severity: Optional[str] = None
    follow_up_questions: List[str] = []
    recommendations: List[str] = []
    conversation_context: Dict[str, Any] = {}

def sanitize_input(text: str) -> str:
    """Sanitize input to prevent injection attacks"""
    return bleach.clean(text, tags=[], attributes={}, strip=True)

async def verify_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> bool:
    """Verify API key or JWT token"""
    # Check API key
    api_key = request.headers.get('X-API-KEY')
    if api_key and api_key == Config.API_KEY:
        return True
    
    # Check JWT token
    if credentials:
        try:
            jwt.decode(credentials.credentials, Config.JWT_SECRET, algorithms=["HS256"])
            return True
        except Exception:
            pass
    
    raise HTTPException(status_code=401, detail="Unauthorized")

@router.post("/analyze-symptom", response_model=SymptomResponse)
@limiter.limit(Config.RATE_LIMIT)
async def analyze_symptom(
    request: Request,
    symptom_request: SymptomRequest,
    auth: bool = Depends(verify_auth)
):
    """Analyze a symptom with caching and async processing"""
    try:
        # Sanitize input
        symptom = sanitize_input(symptom_request.symptom)
        session_id = sanitize_input(symptom_request.session_id) if symptom_request.session_id else None
        
        if not symptom:
            raise HTTPException(status_code=400, detail="Symptom is required")
        
        # Get model manager
        model_manager = await get_model_manager()
        
        # Analyze with caching
        analysis_result = await model_manager.analyze_symptom_cached(
            symptom,
            {'session_id': session_id} if session_id else None
        )
        
        # Sanitize output
        sanitized_result = {
            'symptom_category': sanitize_input(str(analysis_result.get('symptom_category', ''))),
            'confidence': analysis_result.get('confidence', 0.0),
            'severity': sanitize_input(str(analysis_result.get('severity', ''))),
            'follow_up_questions': [sanitize_input(q) for q in analysis_result.get('follow_up_questions', [])],
            'recommendations': [sanitize_input(r) for r in analysis_result.get('recommendations', [])],
            'analysis': analysis_result.get('analysis', {})
        }
        
        # Log (without sensitive data)
        logger.info(f"Symptom analyzed for session: {session_id}")
        
        return SymptomResponse(**sanitized_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing symptom: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/chat", response_model=ChatResponse)
@limiter.limit(Config.RATE_LIMIT)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    auth: bool = Depends(verify_auth)
):
    """Process chat with caching and async processing"""
    try:
        # Sanitize input
        message = sanitize_input(chat_request.message)
        session_id = sanitize_input(chat_request.session_id)
        conversation_history = chat_request.conversation_history or []
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Get model manager
        model_manager = await get_model_manager()
        
        # Process chat with caching
        analysis_result = await model_manager.chat_cached(message, conversation_history)
        
        # Generate response message
        response_message = _generate_chat_response(message, analysis_result)
        
        # Sanitize output
        sanitized_result = {
            'response': sanitize_input(response_message),
            'symptom_category': sanitize_input(str(analysis_result.get('symptom_category', ''))) if analysis_result.get('symptom_category') else None,
            'confidence': analysis_result.get('confidence'),
            'severity': sanitize_input(str(analysis_result.get('severity', ''))) if analysis_result.get('severity') else None,
            'follow_up_questions': [sanitize_input(q) for q in analysis_result.get('follow_up_questions', [])],
            'recommendations': [sanitize_input(r) for r in analysis_result.get('recommendations', [])],
            'conversation_context': analysis_result.get('conversation_context', {})
        }
        
        # Log (without sensitive data)
        logger.info(f"Chat processed for session: {session_id}")
        
        return ChatResponse(**sanitized_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/batch-analyze")
@limiter.limit("5/minute")  # Stricter limit for batch operations
async def batch_analyze_symptoms(
    request: Request,
    symptoms: List[str],
    auth: bool = Depends(verify_auth)
):
    """Batch analyze multiple symptoms for better performance"""
    try:
        if not symptoms or len(symptoms) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Invalid batch size (1-10 symptoms)")
        
        # Sanitize inputs
        sanitized_symptoms = [sanitize_input(s) for s in symptoms]
        
        # Get model manager
        model_manager = await get_model_manager()
        
        # Batch analyze
        results = await model_manager.batch_analyze_symptoms(sanitized_symptoms)
        
        # Sanitize outputs
        sanitized_results = []
        for result in results:
            if result:
                sanitized_result = {
                    'symptom_category': sanitize_input(str(result.get('symptom_category', ''))),
                    'confidence': result.get('confidence', 0.0),
                    'severity': sanitize_input(str(result.get('severity', ''))),
                    'follow_up_questions': [sanitize_input(q) for q in result.get('follow_up_questions', [])],
                    'recommendations': [sanitize_input(r) for r in result.get('recommendations', [])]
                }
                sanitized_results.append(sanitized_result)
            else:
                sanitized_results.append(None)
        
        logger.info(f"Batch analysis completed for {len(symptoms)} symptoms")
        
        return {"results": sanitized_results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

def _generate_chat_response(message: str, analysis_result: Dict) -> str:
    """Generate a natural language response for chat"""
    category = analysis_result.get('symptom_category', 'unknown')
    severity = analysis_result.get('severity', 'unknown')
    questions = analysis_result.get('follow_up_questions', [])
    
    if severity == 'high':
        response = f"I understand you're experiencing {category}. This could be serious. "
        if questions:
            response += f"To better understand your situation, please tell me: {questions[0]}"
    elif severity == 'medium':
        response = f"I see you're dealing with {category}. Let me ask you a few questions to better understand: "
        if questions:
            response += questions[0]
    else:
        response = f"I understand you have {category}. To help you better, could you tell me: "
        if questions:
            response += questions[0]
    
    return response 