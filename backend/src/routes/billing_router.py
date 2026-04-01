"""
Billing and subscription endpoints
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.core.security import get_current_user

router = APIRouter()


@router.get("/subscription")
async def get_subscription(
    org_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get organization subscription details"""
    # TODO: Implement get subscription
    return {
        "plan": "starter",
        "status": "active",
        "renewal_date": "",
        "amount": 499.0,
    }


@router.post("/subscription/upgrade")
async def upgrade_subscription(
    org_id: str,
    new_plan: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upgrade subscription plan"""
    # TODO: Implement subscription upgrade
    return {"plan": new_plan, "status": "upgrading"}


@router.post("/subscription/downgrade")
async def downgrade_subscription(
    org_id: str,
    new_plan: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Downgrade subscription plan"""
    # TODO: Implement subscription downgrade
    return {"plan": new_plan, "status": "downgrading"}


@router.get("/invoices")
async def get_billing_invoices(
    org_id: str,
    skip: int = 0,
    limit: int = 20,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get billing invoices"""
    # TODO: Implement get billing invoices
    return {"invoices": [], "total": 0}


@router.get("/usage")
async def get_usage(
    org_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get API usage and metrics"""
    # TODO: Implement get usage
    return {
        "api_calls": 0,
        "limit": 10000,
        "percentage": 0.0,
        "period": "2024-03",
    }


@router.post("/payment-method")
async def add_payment_method(
    org_id: str,
    payment_data: dict,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Add payment method"""
    # TODO: Implement add payment method (Stripe)
    return {"method_id": "pm_1234", "status": "added"}
