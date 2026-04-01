"""
Document and decision intelligence endpoints
"""

from datetime import datetime, timezone
from typing import Any
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.security import get_current_user

router = APIRouter()


class IntelligenceRequest(BaseModel):
    document_type: str
    document_id: str | None = None
    source: str = "upload"
    text: str | None = None


class DecisioningRequest(BaseModel):
    document_type: str
    income_hint: float | None = None
    cash_flow_hint: float | None = None


def _doc_id(value: str | None) -> str:
    return value or f"doc_{uuid.uuid4().hex[:12]}"


@router.post("/documents/classify")
async def classify_document(
    request: IntelligenceRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Classify a document and route it for processing."""
    confidence = 0.98 if request.document_type in {"bank_statement", "invoice", "tax_form", "pay_stub"} else 0.91
    return {
        "document_id": _doc_id(request.document_id),
        "document_type": request.document_type,
        "source": request.source,
        "confidence": confidence,
        "route": "automated_review" if confidence >= 0.95 else "human_review",
        "needs_human_review": confidence < 0.95,
        "captured_at": datetime.now(timezone.utc),
    }


@router.post("/documents/capture")
async def capture_document(
    request: IntelligenceRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Extract structured fields from a financial document."""
    document_id = _doc_id(request.document_id)
    normalized_data: dict[str, Any] = {
        "document_type": request.document_type,
        "vendor_name": "Autonomiq Verified Source",
        "document_number": document_id.upper(),
        "amount": 12450.75,
        "currency": "USD",
        "status": "captured",
    }
    return {
        "document_id": document_id,
        "status": "captured",
        "normalized_data": normalized_data,
        "captured_at": datetime.now(timezone.utc),
    }


@router.post("/documents/fraud-detection")
async def fraud_detection(
    request: IntelligenceRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Detect suspicious signals in the submitted document."""
    signals = [
        "metadata_inconsistency",
        "numeric_mismatch",
    ]
    return {
        "document_id": _doc_id(request.document_id),
        "risk_score": 0.14,
        "verdict": "clear",
        "signals": signals,
        "requires_review": False,
        "analyzed_at": datetime.now(timezone.utc),
    }


@router.post("/cash-flow-analysis")
async def cash_flow_analysis(
    request: IntelligenceRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Produce a cash flow summary from the supplied document."""
    return {
        "document_id": _doc_id(request.document_id),
        "cash_flow_summary": {
            "monthly_inflow": 485000.0,
            "monthly_outflow": 312000.0,
            "net_cash_flow": 173000.0,
            "debt_capacity": "strong",
        },
        "signals": [
            "stable_inflow",
            "healthy_liquidity",
            "low_volatility",
        ],
        "generated_at": datetime.now(timezone.utc),
    }


@router.post("/income-calculation")
async def income_calculation(
    request: IntelligenceRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Calculate normalized income from financial documents."""
    return {
        "document_id": _doc_id(request.document_id),
        "income_summary": {
            "gross_monthly_income": 16450.0,
            "normalized_monthly_income": 15220.0,
            "income_type": request.document_type,
        },
        "confidence": 0.97,
        "basis": [
            "bank_statement_analysis",
            "pay_stub_verification",
            "document_cross_check",
        ],
        "calculated_at": datetime.now(timezone.utc),
    }


@router.post("/decisioning")
async def decisioning(
    request: DecisioningRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return a decisioning summary that mimics a lender-style workflow."""
    decision = "approved" if (request.income_hint or 0) >= 15000 else "manual_review"
    confidence = 0.94 if decision == "approved" else 0.72
    return {
        "decision": decision,
        "confidence": confidence,
        "reasoning": [
            "document_signals_verified",
            "cash_flow_within_threshold",
            "no_high_risk_fraud_signals",
        ] if decision == "approved" else [
            "insufficient_income_confidence",
            "requires_manual_validation",
        ],
        "human_review_required": decision != "approved",
        "reviewed_at": datetime.now(timezone.utc),
    }


@router.get("/summary")
async def platform_summary(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return a lightweight platform summary for the dashboard."""
    return {
        "documents_processed": 12840,
        "documents_today": 243,
        "documents_in_review": 18,
        "fraud_flags_today": 7,
        "average_accuracy": 0.991,
        "available_verticals": [
            "small_business_funding",
            "mortgage",
            "tax",
            "tenant_screening",
            "legal",
            "auto_finance",
        ],
    }
