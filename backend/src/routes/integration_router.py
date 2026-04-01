"""
Third-party integration endpoints
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.schemas.schemas import IntegrationCreate, IntegrationResponse
from src.core.database import get_db
from src.core.security import get_current_user

router = APIRouter()


@router.post("/", response_model=IntegrationResponse)
async def create_integration(
    integration_data: IntegrationCreate,
    org_id: str = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new integration"""
    # TODO: Implement integration creation
    # 1. Validate integration credentials
    # 2. Test connection
    # 3. Store encrypted credentials
    # 4. Enable data sync
    return {"id": "int_1", "name": integration_data.name, "is_active": False}


@router.get("/", response_model=list[IntegrationResponse])
async def list_integrations(
    org_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List active integrations"""
    # TODO: Implement list integrations
    return []


@router.get("/{integration_id}", response_model=IntegrationResponse)
async def get_integration(
    integration_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get integration details"""
    # TODO: Implement get integration
    return {"id": integration_id, "name": ""}


@router.post("/{integration_id}/sync")
async def sync_integration(
    integration_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Manually trigger data sync"""
    # TODO: Implement sync trigger (Celery task)
    return {"integration_id": integration_id, "status": "syncing"}


@router.post("/{integration_id}/disconnect")
async def disconnect_integration(
    integration_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Disconnect integration"""
    # TODO: Implement disconnection
    return {"integration_id": integration_id, "status": "disconnected"}
