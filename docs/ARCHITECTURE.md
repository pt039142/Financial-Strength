# System Architecture

## Overview

Autonomiq.AI uses a modern, scalable architecture designed to handle enterprise-scale finance operations with AI capabilities.

## Architecture Layers

### 1. Frontend Layer (Next.js)
- **Client-side**: React components with TypeScript
- **Server-side**: Next.js Server Components for SSR
- **Static**: Pre-rendered pages for performance
- **Real-time**: WebSocket support for live updates

### 2. API Layer (FastAPI)
- **REST APIs**: Standard HTTP endpoints
- **WebSocket**: Real-time communication
- **Authentication**: JWT with OAuth2
- **Rate Limiting**: API key-based throttling

### 3. Business Logic Layer (Services)
- **Reconciliation Service**: AI-powered transaction matching
- **Invoice Service**: OCR and data extraction
- **Integration Service**: Third-party API handlers
- **Reporting Service**: Financial analytics

### 4. Data Layer
- **Primary DB**: PostgreSQL with async SQLAlchemy
- **Cache**: Redis for session and query caching
- **File Storage**: AWS S3 for documents
- **Queue**: Redis-based Celery for async tasks

## Data Flow

```
Frontend (React)
     ↓ (HTTPS/REST)
API Gateway (FastAPI)
     ↓
Services Layer
     ├→ Reconciliation Service
     ├→ Invoice Service
     ├→ Integration Service
     └→ Reporting Service
     ↓
Data Layer
     ├→ PostgreSQL
     ├→ Redis
     ├→ AWS S3
     └→ Celery Queue
```

## Core Services

### 1. Authentication Service
```python
- User registration and login
- JWT token generation/refresh
- OAuth2 integration (Google, Microsoft)
- MFA support
- Password reset
```

### 2. Organization Service
```python
- Multi-tenant organization management
- User role-based access control
- Subscription tier management
- API key generation
```

### 3. Finance Data Service
```python
- Transaction management
- Account synchronization
- Data import/export
- Real-time balance tracking
```

### 4. Reconciliation Service
```python
- AI-powered transaction matching
- Confidence scoring
- Batch processing
- Manual approval workflows
```

### 5. Invoice Service
```python
- OCR processing
- Data extraction
- Vendor management
- Approval workflows
```

### 6. Reporting Service
```python
- Financial report generation
- Dashboard data
- Custom report builder
- Export to PDF/Excel
```

### 7. Integration Service
```python
- Third-party API handlers
- Tally, Zoho, QuickBooks
- Data sync scheduling
- Error handling & retry logic
```

### 8. Billing Service
```python
- Subscription management
- Invoice generation
- Payment processing (Stripe)
- Usage tracking
```

## Technology Decisions

### Why FastAPI?
- ✅ Async/await support for high concurrency
- ✅ Automatic API documentation (OpenAPI/Swagger)
- ✅ Type hints with Pydantic validation
- ✅ Fast performance (near Go/Rust speeds)
- ✅ Easy deployment (single file application)

### Why Next.js?
- ✅ Server-side rendering (SSR) for SEO
- ✅ Static site generation (SSG)
- ✅ Incremental static regeneration (ISR)
- ✅ Built-in TypeScript support
- ✅ Image optimization
- ✅ API routes capability

### Why PostgreSQL?
- ✅ ACID compliance for financial data
- ✅ Complex queries with JOINs
- ✅ Full-text search capabilities
- ✅ JSON datatype support
- ✅ Excellent for multi-tenant designs

### Why Redis?
- ✅ Fast in-memory caching
- ✅ Session storage
- ✅ Message broker for Celery
- ✅ Pub/Sub for real-time features
- ✅ Rate limiting

### Why Celery?
- ✅ Asynchronous task processing
- ✅ Long-running operations (reconciliation, invoice processing)
- ✅ Scheduled tasks (data sync, report generation)
- ✅ Task retries and error handling
- ✅ Distributed task execution

## Deployment Architecture

### Development
```
Docker Compose (Local)
├── PostgreSQL Container
├── Redis Container
├── FastAPI Container
├── Celery Worker Container
└── Next.js Container
```

### Production
```
AWS Infrastructure
├── RDS PostgreSQL (Multi-AZ)
├── ElastiCache Redis Cluster
├── ECS Fargate (API + Celery)
├── S3 (File Storage)
├── CloudFront (CDN)
├── ALB (Load Balancer)
└── Route 53 (DNS)
```

## Scalability Considerations

### Horizontal Scaling
- **API**: Multiple container instances behind load balancer
- **Celery**: Multiple worker processes
- **Database**: Read replicas for scaling reads
- **Redis**: Cluster mode for distributed caching

### Vertical Scaling
- **CPU**: For compute-heavy AI operations
- **Memory**: For large dataset processing
- **Storage**: S3 for unlimited file storage

### Database Optimization
- **Indexes**: On frequently queried columns
- **Query optimization**: Avoid N+1 queries
- **Connection pooling**: With SQLAlchemy
- **Read replicas**: For heavy read workloads
- **Partitioning**: For large transaction tables

## Security Architecture

### Authentication & Authorization
```
Frontend
  ↓ (Credentials)
FastAPI → JWT Validation → User Info
  ↓
Protected Routes → Role-Based Access Control
```

### Data Protection
- **Encryption**: TLS for data in transit
- **Hashing**: bcrypt for passwords
- **API Keys**: For service-to-service auth
- **Audit Logs**: All user actions logged
- **Data Encryption**: For sensitive fields at rest

### Rate Limiting
- API key-based throttling
- IP-based rate limiting
- User-based request limits

## Monitoring & Observability

### Logging
- Application logs to CloudWatch/ELK
- Database query logs
- API request/response logs
- Error tracking (Sentry)

### Metrics
- CPU, Memory, Disk usage
- API response times
- Database connection pool
- Celery task processing time
- Redis hit rates

### Alerts
- High error rate (>5%)
- Database connection issues
- Celery queue depth
- API latency spikes

## Disaster Recovery

### Backup Strategy
- Daily PostgreSQL backups to S3
- Point-in-time recovery (35 days)
- Redis persistence to disk
- S3 versioning enabled

### High Availability
- Multi-AZ database deployment
- Redis cluster mode
- Load balanced API instances
- Hot standby for critical services

### Recovery Time Objectives (RTO)
- **Critical Services**: 15 minutes
- **Data**: Less than 1 hour
- **Full System**: Less than 4 hours
