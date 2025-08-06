# Plant Doctor Deployment Guide

This guide will help you deploy your Plant Doctor application following the example structure with ECS Fargate frontend and Lambda backend.

## Architecture Overview

```
User → ALB (port 80) → ECS Frontend Container (port 3000)
                     ↓
Frontend Container → API Gateway → Lambda Function
```

## Deployment Steps

### 1. Deploy Backend (Lambda + API Gateway)

First, deploy your backend using SAM:

```bash
cd backend
sam build
sam deploy
```

This will create:
- Lambda function with API Gateway
- API endpoint at: `https://[api-id].execute-api.us-east-1.amazonaws.com/prod/receive`

### 2. Create ECR Repository for Frontend

```bash
aws ecr create-repository --repository-name plant-doctor-frontend
```

### 3. Build and Push Frontend Docker Image

```bash
cd frontend

# Build the Docker image
docker build -t plant-doctor-frontend .

# Tag the image
docker tag plant-doctor-frontend:latest [YOUR-ACCOUNT-ID].dkr.ecr.us-east-1.amazonaws.com/plant-doctor-frontend:latest

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [YOUR-ACCOUNT-ID].dkr.ecr.us-east-1.amazonaws.com

# Push the image
docker push [YOUR-ACCOUNT-ID].dkr.ecr.us-east-1.amazonaws.com/plant-doctor-frontend:latest
```

### 4. Deploy Frontend Infrastructure

```bash
# Deploy the complete infrastructure (ECS + ALB + VPC)
aws cloudformation deploy --template-file cloudformation.yml --stack-name plant-doctor-infrastructure
```

### 5. Update Frontend API URL

After the backend deployment, get your API Gateway URL and update the frontend:

```bash
# Get the API URL
aws cloudformation describe-stacks --stack-name plant-doctor-backend --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text
```

Then update `frontend/page.tsx` with the new API URL.

### 6. Redeploy Frontend

After updating the API URL, rebuild and push the frontend:

```bash
cd frontend
docker build -t plant-doctor-frontend .
docker tag plant-doctor-frontend:latest [YOUR-ACCOUNT-ID].dkr.ecr.us-east-1.amazonaws.com/plant-doctor-frontend:latest
docker push [YOUR-ACCOUNT-ID].dkr.ecr.us-east-1.amazonaws.com/plant-doctor-frontend:latest
```

### 7. Get Your Application URL

```bash
aws cloudformation describe-stacks --stack-name plant-doctor-infrastructure --query "Stacks[0].Outputs[?OutputKey=='ApplicationURL'].OutputValue" --output text
```

## File Structure

```
PlantDoctor/
├── backend/
│   ├── template.yaml          # SAM template (Lambda + API Gateway)
│   ├── routes.py              # Lambda function code
│   └── ...
├── frontend/
│   ├── page.tsx               # React component
│   ├── Dockerfile             # Frontend container
│   └── ...
├── cloudformation.yml         # Complete infrastructure (ECS + ALB + VPC)
└── DEPLOYMENT_GUIDE.md        # This file
```

## Key Changes Made

1. **Backend**: Changed from Function URL to API Gateway
2. **Frontend**: Containerized for ECS deployment
3. **Infrastructure**: Single CloudFormation file with ECS Fargate + ALB + VPC
4. **API Integration**: Frontend calls API Gateway endpoint

## Troubleshooting

- If ECS tasks fail to start, check the CloudWatch logs
- If ALB health checks fail, ensure your frontend responds on `/` path
- If API calls fail, verify the API Gateway URL is correct 