from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SymptomRequest(BaseModel):
    """Request schema for symptom analysis"""
    symptom: str = Field(..., description="The symptom description", min_length=1, max_length=500)
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation tracking")

class ChatRequest(BaseModel):
    """Request schema for chat endpoint"""
    message: str = Field(..., description="User message", min_length=1, max_length=500)
    session_id: str = Field(..., description="Session ID for conversation tracking")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=[], 
        description="Previous conversation messages"
    )

class SymptomResponse(BaseModel):
    """Response schema for symptom analysis"""
    symptom_category: str = Field(..., description="Detected symptom category")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    severity: str = Field(..., description="Assessed severity level")
    follow_up_questions: List[str] = Field(..., description="Generated follow-up questions")
    recommendations: List[str] = Field(..., description="Medical recommendations")
    analysis: Dict[str, Any] = Field(..., description="Detailed analysis information")

class ChatResponse(BaseModel):
    """Response schema for chat endpoint"""
    response: str = Field(..., description="AI response message")
    symptom_category: Optional[str] = Field(None, description="Detected symptom category")
    confidence: Optional[float] = Field(None, description="Confidence score")
    severity: Optional[str] = Field(None, description="Assessed severity")
    follow_up_questions: List[str] = Field(default=[], description="Follow-up questions")
    recommendations: List[str] = Field(default=[], description="Recommendations")
    conversation_context: Dict[str, Any] = Field(default={}, description="Conversation context")

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    code: Optional[str] = Field(None, description="Error code")

class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="API version")
    model_status: str = Field(..., description="Model loading status") 