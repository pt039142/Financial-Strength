"""
Autonomiq.AI - FastAPI Backend Main Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging

from src.routes import (
    auth_router,
    organization_router,
    finance_router,
    invoice_router,
    intelligence_router,
    reconciliation_router,
    reporting_router,
    integration_router,
    billing_router,
    health_router,
)
from src.core.config import settings
from src.core.database import init_db

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Autonomiq.AI Backend")
    try:
        await init_db()
    except Exception as e:
        logger.warning(f"Database initialization failed (running without database): {e}")
    yield
    # Shutdown
    logger.info("Shutting down Autonomiq.AI Backend")


# Create FastAPI instance
app = FastAPI(
    title="Autonomiq.AI API",
    description="AI-powered Finance Automation Platform API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted Host Middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)


# Include Routers
app.include_router(health_router.router, prefix="/api/v1/health", tags=["Health"])
app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(organization_router.router, prefix="/api/v1/organizations", tags=["Organizations"])
app.include_router(finance_router.router, prefix="/api/v1/finance", tags=["Finance"])
app.include_router(invoice_router.router, prefix="/api/v1/invoices", tags=["Invoices"])
app.include_router(intelligence_router.router, prefix="/api/v1/intelligence", tags=["Intelligence"])
app.include_router(reconciliation_router.router, prefix="/api/v1/reconciliation", tags=["Reconciliation"])
app.include_router(reporting_router.router, prefix="/api/v1/reports", tags=["Reporting"])
app.include_router(integration_router.router, prefix="/api/v1/integrations", tags=["Integrations"])
app.include_router(billing_router.router, prefix="/api/v1/billing", tags=["Billing"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Autonomiq.AI API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
