from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
import logging
from contextlib import asynccontextmanager
from app.models.model_manager import get_model_manager, model_manager
from app.api.routes import router
from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting AI Symptom Checker API...")
    try:
        # Initialize model manager
        await get_model_manager()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Symptom Checker API...")
    try:
        await model_manager.cleanup()
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Create FastAPI app
app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    lifespan=lifespan
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Symptom Checker API is running",
        "version": Config.API_VERSION,
        "device": Config.DEVICE
    }

@app.get("/")
async def index():
    """API information endpoint"""
    return {
        "message": "AI-Powered Symptom Checker API",
        "version": Config.API_VERSION,
        "endpoints": {
            "analyze_symptom": "/api/analyze-symptom",
            "chat": "/api/chat",
            "health": "/health"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=False,  # Disable reload in production
        workers=1,  # Single worker for model sharing
        log_level=Config.LOG_LEVEL.lower()
    ) 