# Autonomiq.AI Backend API

Enterprise-grade FastAPI backend for AI-powered finance automation platform.

## Features

- **FastAPI** - Modern, fast Python web framework
- **Async/Await** - Asynchronous request handling
- **SQLAlchemy 2.0** - ORM with async support
- **PostgreSQL** - Production-grade database
- **Redis** - Caching and task queue
- **Celery** - Asynchronous task processing
- **JWT Authentication** - Secure token-based auth
- **AWS S3** - File storage integration
- **CORS** - Cross-origin resource sharing

## Architecture

```
backend/
├── main.py           # Application entry point
├── requirements.txt  # Python dependencies
├── .env             # Environment variables
└── src/
    ├── core/        # Core configuration, database, security
    ├── routes/      # API endpoint routers
    ├── models/      # SQLAlchemy models
    ├── schemas/     # Pydantic validation schemas
    ├── services/    # Business logic services
    └── utils/       # Utility functions
```

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Environment Setup

Create `.env` file:

```env
DEBUG=True
HOST=0.0.0.0
PORT=8000

DATABASE_URL=postgresql://user:password@localhost:5432/autonomiq_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key-change-in-production

AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_S3_BUCKET=autonomiq-ai-files

SMTP_HOST=smtp.gmail.com
SMTP_USER=your_email
SMTP_PASSWORD=your_password
```

## Running Development Server

```bash
python main.py
```

Server runs at: `http://localhost:8000`

API Docs: `http://localhost:8000/docs`

## API Endpoints

### Authentication
- `POST /api/v1/auth/signup` - Register new user
- `POST /api/v1/auth/login` - Login user
- `POST /api/v1/auth/refresh` - Refresh access token
- `POST /api/v1/auth/logout` - Logout user

### Organizations
- `POST /api/v1/organizations/` - Create organization
- `GET /api/v1/organizations/` - List organizations
- `GET /api/v1/organizations/{org_id}` - Get organization details

### Finance Data
- `GET /api/v1/finance/accounts` - Get accounts
- `GET /api/v1/finance/transactions` - Get transactions
- `POST /api/v1/finance/transactions/import` - Import transactions

### Invoices
- `POST /api/v1/invoices/upload` - Upload invoice
- `GET /api/v1/invoices/` - List invoices
- `POST /api/v1/invoices/extract` - Extract invoice data

### Reconciliation
- `POST /api/v1/reconciliation/` - Reconcile transactions
- `POST /api/v1/reconciliation/{match_id}/approve` - Approve match
- `POST /api/v1/reconciliation/{match_id}/reject` - Reject match

### Reporting
- `POST /api/v1/reports/` - Generate report
- `GET /api/v1/reports/` - List reports
- `POST /api/v1/reports/{report_id}/export` - Export report

### Integrations
- `POST /api/v1/integrations/` - Create integration
- `GET /api/v1/integrations/` - List integrations
- `POST /api/v1/integrations/{id}/sync` - Sync data

### Billing
- `GET /api/v1/billing/subscription` - Get subscription
- `POST /api/v1/billing/subscription/upgrade` - Upgrade plan
- `GET /api/v1/billing/invoices` - Get billing invoices
- `GET /api/v1/billing/usage` - Get API usage

## Database Setup

```bash
# Create database
createdb autonomiq_db

# Run migrations (if using Alembic)
alembic upgrade head
```

## Running with Docker

```bash
docker build -t autonomiq-api .
docker run -p 8000:8000 autonomiq-api
```

## Testing

```bash
pytest tests/
```

## Deployment

### Docker Compose
```bash
docker-compose up
```

### Production with Gunicorn
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

## Documentation

API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Support

For issues or questions, contact: support@autonomiq.ai
