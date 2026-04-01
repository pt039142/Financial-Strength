# Development Guide

## Development Environment Setup

### System Requirements

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose
- Git

### Initial Setup

#### 1. Clone Repository

```bash
git clone https://github.com/autonomiq/autonomiq-ai.git
cd autonomiq-ai
```

#### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Edit .env with your local settings (especially DATABASE_URL)
```

#### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env.local if needed
cp .env.example .env.local
```

### Quick Start with Docker Compose

```bash
docker-compose up

# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## Development Workflow

### Branch Strategy

We follow Git Flow branching model:

```
main (production)
  ↑
release/v1.0.0
  ↑
develop (staging)
  ↓
feature/user-auth
feature/invoice-processing
bugfix/login-issue
```

### Creating a Feature

```bash
# Update develop branch
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add your feature description"

# Push to remote
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

Examples:
```
feat(auth): add JWT token refresh endpoint
fix(reconciliation): correct transaction matching algorithm
docs(api): update endpoint documentation
```

## Backend Development

### Project Structure

```
backend/
├── main.py              # Entry point
├── requirements.txt     # Dependencies
├── .env.example         # Environment template
├── Dockerfile           # Container definition
├── src/
│   ├── core/           # Configuration & setup
│   ├── routes/         # API endpoints
│   ├── models/         # Database models
│   ├── schemas/        # Pydantic validators
│   ├── services/       # Business logic
│   └── utils/          # Helper functions
└── tests/              # Test files
```

### Running Backend

```bash
# Development server with hot reload
python main.py

# Production with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

### Database Migrations

```bash
# Initialize Alembic (one-time)
alembic init migrations

# Create migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Creating a New API Endpoint

1. **Create Pydantic Schema** in `src/schemas/schemas.py`:

```python
class MyResourceCreate(BaseModel):
    name: str
    description: Optional[str] = None

class MyResourceResponse(MyResourceCreate):
    id: str
    created_at: datetime
```

2. **Create Database Model** in `src/models/models.py`:

```python
class MyResource(Base):
    __tablename__ = "my_resources"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
```

3. **Create Service** in `src/services/my_service.py`:

```python
class MyService:
    @staticmethod
    async def create(org_id: str, data: dict, db: AsyncSession):
        # Business logic here
        pass
```

4. **Create Router** in `src/routes/my_router.py`:

```python
from fastapi import APIRouter, Depends
from src.core.database import get_db
from src.core.security import get_current_user
from src.schemas.schemas import MyResourceCreate, MyResourceResponse

router = APIRouter()

@router.post("/", response_model=MyResourceResponse)
async def create_resource(
    data: MyResourceCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Implementation
    pass
```

5. **Include Router** in `main.py`:

```python
from src.routes import my_router

app.include_router(
    my_router.router,
    prefix="/api/v1/my-resources",
    tags=["My Resources"]
)
```

### Writing Tests

```python
# tests/test_my_router.py
import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_create_resource():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/my-resources/",
            json={"name": "Test Resource"},
            headers={"Authorization": "Bearer test-token"}
        )
        assert response.status_code == 201
```

Run tests:

```bash
pytest tests/
pytest tests/test_my_router.py -v  # Specific test file
pytest tests/test_my_router.py::test_create_resource -v  # Specific test
```

## Frontend Development

### Project Structure

```
frontend/
├── src/
│   ├── app/            # Next.js app directory
│   │   ├── page.tsx
│   │   ├── layout.tsx
│   │   └── [routes]/
│   ├── components/     # Reusable components
│   ├── lib/           # Utilities & hooks
│   └── styles/        # Global styles
├── public/            # Static assets
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.js
```

### Running Frontend

```bash
# Development server
npm run dev

# Production build
npm run build
npm start

# Production build locally
npm run build && npm start
```

### Creating a New Page

1. **Create page file** in `src/app/your-page/page.tsx`:

```typescript
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Your Page - Autonomiq.AI",
  description: "Page description",
};

export default function YourPage() {
  return (
    <div className="pt-32 pb-24">
      <div className="container mx-auto px-4">
        <h1 className="text-5xl font-bold">Your Page</h1>
        {/* Page content */}
      </div>
    </div>
  );
}
```

2. **Add to navigation** in `src/components/Navbar.tsx`:

```typescript
const navLinks = [
  // ...
  { href: '/your-page', label: 'Your Page' },
];
```

### Creating a Component

```typescript
// src/components/MyComponent.tsx
'use client';

import { useState } from 'react';

interface MyComponentProps {
  title: string;
  onClick?: () => void;
}

export function MyComponent({ title, onClick }: MyComponentProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="...">
      {/* Component content */}
    </div>
  );
}
```

### Using Tailwind CSS

Common classes:

```typescript
// Spacing
className="p-4 m-2 mb-4"

// Colors
className="text-primary bg-dark-secondary border-secondary"

// Responsive
className="text-sm md:text-lg lg:text-2xl"

// Flexbox
className="flex items-center justify-between gap-4"

// Grid
className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"

// Hover states
className="hover:bg-primary transition"

// Animations
className="animate-fade-in"
```

### TypeScript Best Practices

```typescript
// Use interfaces for component props
interface UserProps {
  id: string;
  name: string;
  email: string;
  isActive?: boolean;  // Optional property
}

// Use types for API responses
type ApiResponse<T> = {
  data: T;
  error?: string;
  success: boolean;
};

// Avoid using 'any'
// Use 'unknown' and type guards instead
function processData(data: unknown) {
  if (typeof data === 'string') {
    // data is narrowed to string here
  }
}
```

## Code Quality Standards

### Linting

```bash
# Backend
cd backend
pylint src/
flake8 src/

# Frontend
cd frontend
npm run lint
```

### Formatting

```bash
# Backend
python -m black src/
python -m isort src/

# Frontend
npm run format
```

### Type Checking

```bash
# Frontend
npm run type-check

# Backend
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Debugging

### Backend Debugging

Using VS Code:

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["main:app", "--reload"],
      "jinja": true,
      "justMyCode": true
    }
  ]
}
```

### Frontend Debugging

Use VS Code built-in debugging:

1. Set breakpoint in code
2. Click "Debug" in VS Code
3. Breakpoints will pause execution

## Performance Optimization

### Backend

```python
# Use database indexes
class User(Base):
    __tablename__ = "users"
    email = Column(String, index=True)  # Add index

# Eager load relationships
users = db.query(User).options(joinedload(User.organization))

# Use pagination
users = db.query(User).offset(skip).limit(limit)

# Cache frequently accessed data
@cache.cached(timeout=300)
def get_popular_items():
    pass
```

### Frontend

```typescript
// Use lazy loading for images
<Image src="..." alt="..." loading="lazy" />

// Code split routes
const Dashboard = dynamic(() => import('./Dashboard'), { 
  loading: () => <Skeleton />
})

// Memoize components
const MyComponent = memo(({ prop }) => {
  return <div>{prop}</div>
})

// Use SWR for data fetching
import useSWR from 'swr'
const { data } = useSWR('/api/data', fetcher)
```

## Documentation

### Adding Code Comments

```python
def calculate_reconciliation_score(
    transaction1: Transaction,
    transaction2: Transaction,
) -> float:
    """
    Calculate similarity score between two transactions.
    
    Uses a weighted algorithm considering amount, date, and description.
    
    Args:
        transaction1: First transaction to compare
        transaction2: Second transaction to compare
    
    Returns:
        float: Similarity score between 0.0 and 1.0
    
    Example:
        >>> tx1 = Transaction(amount=100.0)
        >>> tx2 = Transaction(amount=100.0)
        >>> score = calculate_reconciliation_score(tx1, tx2)
        >>> assert score > 0.9
    """
    # Implementation
    pass
```

### Updating API Docs

OpenAPI docs are automatically generated from FastAPI docstrings:

```python
@router.post("/reconcile")
async def reconcile_transactions(request: ReconciliationRequest):
    """
    Reconcile bank and ledger transactions.
    
    This endpoint uses AI-powered matching to identify related transactions
    between bank and ledger records.
    
    Args:
        request: Contain bank and ledger transactions
    
    Returns:
        ReconciliationResponse containing matches and statistics
    
    Raises:
        HTTPException: If reconciliation fails
    """
    pass
```

## Problem Solving

### Common Issues

**Issue**: Database connection fails
```bash
# Check PostgreSQL is running
psql -U autonomiq_user -h localhost -d autonomiq_db

# Check connection string in .env
echo $DATABASE_URL
```

**Issue**: Redis connection fails
```bash
# Check Redis is running
redis-cli PING

# Check Redis URL
echo $REDIS_URL
```

**Issue**: Frontend build fails
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

## Getting Help

- **GitHub Issues**: Report bugs and request features
- **Discord**: Community support and discussions
- **Email**: dev-support@autonomiq.ai
- **Documentation**: https://docs.autonomiq.ai
