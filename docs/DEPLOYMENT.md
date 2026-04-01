# Deployment Guide

## Prerequisites

- AWS Account with appropriate permissions
- Docker installed locally
- AWS CLI configured
- kubectl installed for Kubernetes deployments
- Git

## Local Development Deployment

### Using Docker Compose

```bash
# Clone repository
git clone https://github.com/autonomiq/autonomiq-ai.git
cd autonomiq-ai

# Create environment file
cp backend/.env.example backend/.env
# Edit backend/.env with your local settings

# Start all services
docker-compose up

# Verify services
curl http://localhost:8000/api/v1/health
open http://localhost:3000
```

### Manual Local Setup

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup database
createdb autonomiq_db
export DATABASE_URL="postgresql://localhost/autonomiq_db"

# Run migrations (when available)
alembic upgrade head

# Start server
python main.py
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## AWS Production Deployment

### Step 1: Prepare AWS Infrastructure

#### Create RDS PostgreSQL

```bash
aws rds create-db-instance \
  --db-instance-identifier autonomiq-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 15.3 \
  --master-username admin \
  --master-user-password your-secure-password \
  --allocated-storage 100 \
  --backup-retention-period 30 \
  --multi-az
```

#### Create ElastiCache Redis

```bash
aws elasticache create-cache-cluster \
  --cache-cluster-id autonomiq-redis \
  --cache-node-type cache.t3.micro \
  --engine redis \
  --engine-version 7.0 \
  --num-cache-nodes 1
```

#### Create S3 Bucket

```bash
aws s3 mb s3://autonomiq-ai-files --region us-east-1
aws s3api put-bucket-versioning \
  --bucket autonomiq-ai-files \
  --versioning-configuration Status=Enabled
```

#### Configure VPC Security Groups

```bash
# Allow RDS access from ECS
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol tcp \
  --port 5432 \
  --source-security-group-id sg-yyy

# Allow Redis access from ECS
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol tcp \
  --port 6379 \
  --source-security-group-id sg-yyy
```

### Step 2: Build and Push Docker Images

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Create ECR repositories
aws ecr create-repository --repository-name autonomiq-api
aws ecr create-repository --repository-name autonomiq-web

# Build and push API image
docker build -t autonomiq-api:latest ./backend
docker tag autonomiq-api:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/autonomiq-api:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/autonomiq-api:latest

# Build and push Frontend image
docker build -t autonomiq-web:latest ./frontend
docker tag autonomiq-web:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/autonomiq-web:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/autonomiq-web:latest
```

### Step 3: Deploy with ECS Fargate

#### Create ECS Cluster

```bash
aws ecs create-cluster --cluster-name autonomiq-cluster
```

#### Register Task Definitions

Create `ecs-task-api.json`:

```json
{
  "family": "autonomiq-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "autonomiq-api",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/autonomiq-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint:5432/autonomiq_db"
        },
        {
          "name": "REDIS_URL",
          "value": "redis://redis-endpoint:6379/0"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/autonomiq-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Register the task:

```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-api.json
```

#### Create ECS Service

```bash
aws ecs create-service \
  --cluster autonomiq-cluster \
  --service-name autonomiq-api \
  --task-definition autonomiq-api \
  --desired-count 2 \
  --launch-type FARGATE \
  --load-balancers targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=autonomiq-api,containerPort=8000 \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Step 4: Set Up Load Balancer

```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name autonomiq-alb \
  --subnets subnet-xxx subnet-yyy

# Create target group
aws elbv2 create-target-group \
  --name autonomiq-api-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-xxx

# Create listener
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:... \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...
```

### Step 5: Configure CloudFront CDN

```bash
# Create CloudFront distribution
aws cloudfront create-distribution \
  --distribution-config file://cloudfront-config.json
```

### Step 6: Setup Route 53 DNS

```bash
# Create hosted zone
aws route53 create-hosted-zone \
  --name autonomiq.ai \
  --caller-reference date +%s

# Create A record
aws route53 change-resource-record-sets \
  --hosted-zone-id ZXXXXX \
  --change-batch file://route53-change.json
```

## Database Migrations

### Using Alembic

```bash
# Create migration
alembic revision --autogenerate -m "Add new column"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Environment Configuration

### Production Environment Variables

```bash
# Backend
export DEBUG=False
export SECRET_KEY=your-long-random-secret-key
export DATABASE_URL=postgresql://user:pass@rds-endpoint:5432/autonomiq_db
export REDIS_URL=redis://redis-endpoint:6379/0
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_S3_BUCKET=autonomiq-ai-files
export CELERY_BROKER_URL=redis://redis-endpoint:6379/1
export CELERY_RESULT_BACKEND=redis://redis-endpoint:6379/2

# Frontend
export NEXT_PUBLIC_API_URL=https://api.autonomiq.ai
```

## Monitoring & Logging

### Setup CloudWatch

```bash
# Create log group
aws logs create-log-group --log-group-name /ecs/autonomiq

# Create alarms
aws cloudwatch put-metric-alarm \
  --alarm-name autonomiq-high-error-rate \
  --alarm-description "Alert on high error rate" \
  --metric-name ErrorCount \
  --namespace AWS/ECS \
  --statistic Sum \
  --period 300 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold
```

### Setup CloudTrail

```bash
aws cloudtrail create-trail \
  --name autonomiq-trail \
  --s3-bucket autonomiq-cloudtrail-logs

aws cloudtrail start-logging \
  --trail-name autonomiq-trail
```

## SSL/TLS Certificate

### Using AWS Certificate Manager

```bash
aws acm request-certificate \
  --domain-name autonomiq.ai \
  --subject-alternative-names www.autonomiq.ai \
  --validation-method DNS
```

## Scaling Configuration

### Auto-scaling Policy

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/autonomiq-cluster/autonomiq-api \
  --min-capacity 2 \
  --max-capacity 10

# Create scaling policy
aws application-autoscaling put-scaling-policy \
  --policy-name autonomiq-scaling \
  --service-namespace ecs \
  --resource-id service/autonomiq-cluster/autonomiq-api \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

## Backup & Recovery

### Database Backups

```bash
# Create RDS snapshot
aws rds create-db-snapshot \
  --db-instance-identifier autonomiq-db \
  --db-snapshot-identifier autonomiq-backup-$(date +%Y%m%d)

# Restore from snapshot
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier autonomiq-db-restored \
  --db-snapshot-identifier autonomiq-backup-20240101
```

### S3 Backup

```bash
# Enable versioning
aws s3api put-bucket-versioning \
  --bucket autonomiq-ai-files \
  --versioning-configuration Status=Enabled

# Setup lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
  --bucket autonomiq-ai-files \
  --lifecycle-configuration file://s3-lifecycle.json
```

## Troubleshooting

### Check ECS Service Status

```bash
aws ecs describe-services \
  --cluster autonomiq-cluster \
  --services autonomiq-api
```

### View Logs

```bash
aws logs tail /ecs/autonomiq-api --follow
```

### Test Database Connection

```bash
psql -h rds-endpoint -U admin -d autonomiq_db
```

### Test Redis Connection

```bash
redis-cli -h redis-endpoint PING
```

## Support

For deployment issues, contact: devops@autonomiq.ai
