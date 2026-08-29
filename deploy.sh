#!/bin/bash

# LucilleLLM Deployment Script
echo "🚀 Deploying LucilleLLM to Google Cloud Run..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK first."
    exit 1
fi

# Check if project is set
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No Google Cloud project set. Please run:"
    echo "   gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "📋 Using project: $PROJECT_ID"

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Create Secret Manager secret for OpenAI API key
echo "🔐 Setting up secrets..."
if [ -f ".env" ]; then
    OPENAI_KEY=$(grep OPENAI_API_KEY .env | cut -d '=' -f2)
    if [ ! -z "$OPENAI_KEY" ]; then
        echo "Creating OpenAI API key secret..."
        echo -n "$OPENAI_KEY" | gcloud secrets create openai-api-key --data-file=- --replication-policy=automatic || echo "Secret already exists"
    else
        echo "⚠️  OPENAI_API_KEY not found in .env file"
    fi
else
    echo "⚠️  .env file not found. Please create it with your OpenAI API key."
fi

# Deploy with Cloud Build
echo "🏗️  Building and deploying..."
gcloud builds submit --config cloudbuild.yaml

echo "✅ Deployment complete!"
echo "🌐 Your app will be available at:"
# Must match the RUNNING production service: lucille / us-central1.
# Neither the original (lucillellm / us-central1) nor cloudbuild's old target
# (lucillellm2 / us-east4) exists in this project — both returned NOT_FOUND.
gcloud run services describe lucille --region=us-central1 --format="value(status.url)"
