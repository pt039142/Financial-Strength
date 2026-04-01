# API Reference

## Base URL

```
https://api.autonomiq.ai/api/v1
```

## Authentication

All API endpoints (except `/auth/signup` and `/auth/login`) require JWT authentication.

### Request Headers

```http
Authorization: Bearer <access_token>
Content-Type: application/json
```

### Token Response

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

## Status Codes

| Code | Meaning |
|------|---------|
| 200 | OK |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 409 | Conflict |
| 422 | Unprocessable Entity |
| 429 | Too Many Requests |
| 500 | Internal Server Error |

## Endpoints

### Authentication

#### Sign Up
```http
POST /auth/signup
Content-Type: application/json

{
  "email": "user@company.com",
  "password": "secure_password",
  "first_name": "John",
  "last_name": "Doe",
  "company_name": "Acme Corp"
}
```

Response:
```json
{
  "access_token": "...",
  "refresh_token": "...",
  "token_type": "bearer"
}
```

#### Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@company.com",
  "password": "secure_password"
}
```

#### Refresh Token
```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "..."
}
```

### Organizations

#### Create Organization
```http
POST /organizations/
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Acme Corporation",
  "description": "Leading logistics company"
}
```

Response:
```json
{
  "id": "org_123",
  "name": "Acme Corporation",
  "slug": "acme-corporation",
  "subscription_tier": "starter",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### List Organizations
```http
GET /organizations/
Authorization: Bearer <token>
```

#### Get Organization
```http
GET /organizations/{org_id}
Authorization: Bearer <token>
```

### Finance Data

#### Get Accounts
```http
GET /finance/accounts?org_id=org_123
Authorization: Bearer <token>
```

Response:
```json
{
  "accounts": [
    {
      "id": "acc_123",
      "name": "Operating Account",
      "type": "checking",
      "balance": 50000.00,
      "currency": "USD"
    }
  ]
}
```

#### Get Transactions
```http
GET /finance/transactions?org_id=org_123&skip=0&limit=100
Authorization: Bearer <token>
```

Response:
```json
{
  "transactions": [
    {
      "id": "txn_123",
      "account": "acc_123",
      "date": "2024-01-15T10:30:00Z",
      "description": "Payment to vendor",
      "amount": 1000.00,
      "currency": "USD",
      "status": "matched"
    }
  ],
  "total": 245
}
```

#### Import Transactions
```http
POST /finance/transactions/import
Authorization: Bearer <token>
Content-Type: application/json

{
  "org_id": "org_123",
  "transactions": [
    {
      "date": "2024-01-15",
      "description": "Payment",
      "amount": 1000.00,
      "account": "acc_123"
    }
  ]
}
```

### Invoices

#### Upload Invoice
```http
POST /invoices/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <invoice.pdf>
org_id: org_123
```

Response:
```json
{
  "invoice_id": "inv_123",
  "status": "processing"
}
```

#### Extract Invoice Data
```http
POST /invoices/{invoice_id}/extract
Authorization: Bearer <token>

{}
```

Response:
```json
{
  "vendor_name": "ABC Supplies Inc",
  "invoice_number": "INV-2024-001",
  "amount": 5000.00,
  "currency": "USD",
  "invoice_date": "2024-01-10",
  "due_date": "2024-02-10",
  "confidence": 0.95
}
```

#### List Invoices
```http
GET /invoices/?org_id=org_123&skip=0&limit=50
Authorization: Bearer <token>
```

### Reconciliation

#### Reconcile Transactions
```http
POST /reconciliation/
Authorization: Bearer <token>
Content-Type: application/json

{
  "org_id": "org_123",
  "bank_transactions": [
    {
      "id": "bank_1",
      "date": "2024-01-15T10:00:00Z",
      "amount": 1000.00,
      "description": "Payment in",
      "account": "acc_123"
    }
  ],
  "ledger_transactions": [
    {
      "id": "ledger_1",
      "date": "2024-01-15T10:00:00Z",
      "amount": 1000.00,
      "description": "Customer payment",
      "account": "acc_123"
    }
  ]
}
```

Response:
```json
{
  "matches": [
    {
      "bank_transaction_id": "bank_1",
      "ledger_transaction_id": "ledger_1",
      "confidence": 0.98,
      "amount_difference": 0.00
    }
  ],
  "unmatched_bank": 0,
  "unmatched_ledger": 0,
  "total_matched_amount": 1000.00
}
```

#### Approve Match
```http
POST /reconciliation/{match_id}/approve
Authorization: Bearer <token>
```

#### Reject Match
```http
POST /reconciliation/{match_id}/reject
Authorization: Bearer <token>
Content-Type: application/json

{
  "reason": "Duplicate entry"
}
```

### Reporting

#### Generate Report
```http
POST /reports/
Authorization: Bearer <token>
Content-Type: application/json

{
  "org_id": "org_123",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-31T23:59:59Z",
  "report_type": "balance_sheet"
}
```

Response:
```json
{
  "id": "report_123",
  "report_type": "balance_sheet",
  "data": {
    "assets": 100000.00,
    "liabilities": 50000.00,
    "equity": 50000.00
  },
  "generated_at": "2024-01-31T10:30:00Z"
}
```

#### Get Dashboard Data
```http
GET /reports/{org_id}/dashboard
Authorization: Bearer <token>
```

Response:
```json
{
  "total_transactions": 1250,
  "reconciliation_rate": 98.5,
  "pending_invoices": 12,
  "monthly_revenue": 250000.00
}
```

### Integrations

#### Create Integration
```http
POST /integrations/
Authorization: Bearer <token>
Content-Type: application/json

{
  "org_id": "org_123",
  "name": "tally",
  "api_key": "your_api_key",
  "api_secret": "your_api_secret"
}
```

#### List Integrations
```http
GET /integrations/?org_id=org_123
Authorization: Bearer <token>
```

#### Sync Integration
```http
POST /integrations/{integration_id}/sync
Authorization: Bearer <token>
```

Response:
```json
{
  "integration_id": "int_123",
  "status": "syncing",
  "job_id": "job_456"
}
```

### Billing

#### Get Subscription
```http
GET /billing/subscription?org_id=org_123
Authorization: Bearer <token>
```

Response:
```json
{
  "plan": "growth",
  "status": "active",
  "renewal_date": "2024-02-15",
  "amount": 1499.00,
  "next_billing_amount": 1499.00
}
```

#### Upgrade Plan
```http
POST /billing/subscription/upgrade
Authorization: Bearer <token>
Content-Type: application/json

{
  "org_id": "org_123",
  "new_plan": "enterprise"
}
```

#### Get Usage
```http
GET /billing/usage?org_id=org_123
Authorization: Bearer <token>
```

Response:
```json
{
  "api_calls": 4500,
  "limit": 10000,
  "percentage": 45,
  "period": "2024-02"
}
```

## Rate Limiting

API requests are rate-limited based on subscription tier:

| Plan | Requests/Min | Requests/Month |
|------|-------------|---|
| Starter | 100 | 10,000 |
| Growth | 1,000 | 100,000 |
| Enterprise | Unlimited | Unlimited |

Rate limit info is included in response headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1609372800
```

## Error Responses

### Format

```json
{
  "detail": "Error message",
  "status": 400,
  "type": "validation_error",
  "errors": [
    {
      "field": "email",
      "message": "Invalid email format"
    }
  ]
}
```

### Common Errors

| Code | Message | Description |
|------|---------|---|
| 401 | "Invalid authentication credentials" | Token missing or invalid |
| 403 | "Insufficient permissions" | User lacks required role |
| 422 | "Validation failed" | Request data validation failed |
| 429 | "Rate limit exceeded" | Too many requests |

## Webhooks

### Supported Events

- `invoice.processed`
- `reconciliation.completed`
- `transaction.flagged`
- `report.generated`

### Register Webhook

```http
POST /webhooks/
Authorization: Bearer <token>
Content-Type: application/json

{
  "url": "https://your-server.com/webhook",
  "events": ["invoice.processed"],
  "active": true
}
```

## Pagination

Add pagination parameters:

```http
GET /invoices/?skip=0&limit=50
```

Responses include:

```json
{
  "data": [...],
  "total": 250,
  "skip": 0,
  "limit": 50
}
```

## Sorting

Sort results:

```http
GET /transactions/?sort_by=date&order=desc
```

## Filtering

Filter results:

```http
GET /invoices/?status=pending&start_date=2024-01-01
```

## Support

API Documentation: https://api.autonomiq.ai/docs
Support Email: api-support@autonomiq.ai
