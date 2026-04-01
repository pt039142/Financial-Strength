"""
Health check endpoints
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "autonomiq-ai-api",
        "version": "1.0.0",
    }


@router.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check endpoint"""
    return {
        "status": "ready",
        "service": "autonomiq-ai-api",
    }
