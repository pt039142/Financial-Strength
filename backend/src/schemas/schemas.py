"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime


# Auth Schemas
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    company_name: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: str
    is_active: bool
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


# Organization Schemas
class OrganizationBase(BaseModel):
    name: str
    description: Optional[str] = None


class OrganizationCreate(OrganizationBase):
    pass


class OrganizationResponse(OrganizationBase):
    id: str
    slug: str
    subscription_tier: str
    created_at: datetime

    class Config:
        from_attributes = True


# Transaction Schemas
class TransactionBase(BaseModel):
    account: str
    amount: float
    description: str
    currency: str = "USD"


class TransactionCreate(TransactionBase):
    date: datetime
    type: str  # bank, ledger


class TransactionResponse(TransactionCreate):
    id: str
    status: str
    ai_confidence: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


# Invoice Schemas
class InvoiceBase(BaseModel):
    vendor_name: str
    invoice_number: str
    amount: float
    currency: str = "USD"


class InvoiceCreate(InvoiceBase):
    invoice_date: datetime
    due_date: datetime


class InvoiceResponse(InvoiceCreate):
    id: str
    status: str
    ocr_confidence: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


# Reconciliation Schemas
class ReconciliationRequest(BaseModel):
    bank_transactions: List[TransactionCreate]
    ledger_transactions: List[TransactionCreate]


class ReconciliationMatch(BaseModel):
    bank_transaction_id: str
    ledger_transaction_id: str
    confidence: float
    amount_difference: float


class ReconciliationResponse(BaseModel):
    matches: List[ReconciliationMatch]
    unmatched_bank: int
    unmatched_ledger: int
    total_matched_amount: float


# Integration Schemas
class IntegrationBase(BaseModel):
    name: str  # tally, zoho, quickbooks


class IntegrationCreate(IntegrationBase):
    api_key: str
    api_secret: Optional[str] = None


class IntegrationResponse(IntegrationBase):
    id: str
    is_active: bool
    last_synced: Optional[datetime]

    class Config:
        from_attributes = True


# Report Schemas
class FinancialReportRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    report_type: str  # balance_sheet, income_statement, cash_flow


class FinancialReportResponse(BaseModel):
    id: str
    report_type: str
    data: dict
    generated_at: datetime
