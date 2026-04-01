"""
Models for Autonomiq.AI database
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Enum, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from src.core.database import Base


class UserRole(str, enum.Enum):
    """User roles"""
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    VIEWER = "viewer"


class TransactionStatus(str, enum.Enum):
    """Transaction reconciliation status"""
    PENDING = "pending"
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    FLAGGED = "flagged"


class User(Base):
    """User model"""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    first_name = Column(String)
    last_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    role = Column(String, default=UserRole.USER)
    organization_id = Column(String, ForeignKey("organizations.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    organization = relationship("Organization", back_populates="users")


class Organization(Base):
    """Organization model"""
    __tablename__ = "organizations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, index=True)
    description = Column(Text)
    logo_url = Column(String)
    website = Column(String)
    industry = Column(String)
    size = Column(String)  # small, medium, large, enterprise
    is_active = Column(Boolean, default=True)
    subscription_tier = Column(String, default="starter")  # starter, growth, enterprise
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    users = relationship("User", back_populates="organization")
    integrations = relationship("Integration", back_populates="organization")


class Integration(Base):
    """Third-party integration model"""
    __tablename__ = "integrations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    name = Column(String, nullable=False)  # tally, zoho, quickbooks, xero
    api_key = Column(String)  # encrypted
    api_secret = Column(String)  # encrypted
    access_token = Column(String)  # encrypted
    refresh_token = Column(String)  # encrypted
    is_active = Column(Boolean, default=True)
    last_synced = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    organization = relationship("Organization", back_populates="integrations")


class Transaction(Base):
    """Bank/Ledger transaction model"""
    __tablename__ = "transactions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    type = Column(String)  # bank, ledger
    account = Column(String, index=True)
    date = Column(DateTime, index=True)
    description = Column(String)
    amount = Column(Float)
    currency = Column(String, default="USD")
    status = Column(String, default=TransactionStatus.PENDING)
    matched_with = Column(String)  # transaction ID it matched with
    ai_confidence = Column(Float)  # 0-1 confidence score
    source = Column(String)  # integration name
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Invoice(Base):
    """Invoice model"""
    __tablename__ = "invoices"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    file_path = Column(String)
    s3_key = Column(String)
    vendor_name = Column(String, index=True)
    vendor_id = Column(String)
    invoice_number = Column(String, unique=True)
    invoice_date = Column(DateTime)
    due_date = Column(DateTime)
    amount = Column(Float)
    currency = Column(String, default="USD")
    status = Column(String, default="pending")  # pending, processed, rejected, paid
    extracted_data = Column(Text)  # JSON
    ocr_confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(Base):
    """Audit log for compliance tracking"""
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"))
    action = Column(String, index=True)
    entity_type = Column(String)
    entity_id = Column(String)
    changes = Column(Text)  # JSON
    ip_address = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
