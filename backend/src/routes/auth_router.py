"""
Authentication endpoints
"""

from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.schemas.schemas import (
    LoginRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
)
from src.core.database import get_db
from src.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
)

router = APIRouter()


@router.post("/signup", response_model=TokenResponse)
async def signup(
    request: SignupRequest,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user"""
    # TODO: Implement user registration
    # Check if user exists
    # Create organization
    # Create user
    # Return tokens
    return {
        "access_token": "token...",
        "refresh_token": "refresh...",
        "token_type": "bearer",
    }


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Login user and return tokens"""
    # TODO: Implement login
    # Find user by email
    # Verify password
    # Generate tokens
    # Return tokens
    return {
        "access_token": "token...",
        "refresh_token": "refresh...",
        "token_type": "bearer",
    }


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    # TODO: Implement token refresh
    # Verify refresh token
    # Generate new access token
    return {
        "access_token": "new_token...",
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@router.post("/logout")
async def logout():
    """Logout user"""
    # TODO: Implement logout (invalidate tokens)
    return {"message": "Logged out successfully"}
