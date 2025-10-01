# 🚀 LucilleLLM Deployment Guide

## Prerequisites

1. **Google Cloud SDK** installed and configured
2. **OpenAI API Key** from OpenAI platform
3. **Google Cloud Project** with billing enabled

## Step-by-Step Deployment

### 1. Setup Google Cloud Project

```bash
# Set your project (replace with your actual project ID)
gcloud config set project YOUR_PROJECT_ID

# Verify you're authenticated
gcloud auth list
```

### 2. Prepare Environment Variables

Create a `.env` file in your project root:

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual values
OPENAI_API_KEY=sk-your-openai-api-key-here
GOOGLE_CLOUD_PROJECT=your-project-id
ENVIRONMENT=production
```

### 3. Deploy Using the Script

```bash
# Run the automated deployment script
./deploy.sh
```

**OR manually step by step:**

```bash
# Enable APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Create secret for OpenAI API key
echo -n "your-openai-api-key" | gcloud secrets create openai-api-key --data-file=-

# Deploy
gcloud builds submit --config cloudbuild.yaml
```

### 4. Verify Deployment

After deployment, you'll get a URL like:
```
https://lucillellm-xxxxxxxxx-uc.a.run.app
```

Test the endpoints:
- `GET /` - Get session ID
- `POST /chat` - Send messages to Lucille
- `GET /chat/{session_id}` - Get chat history

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes |
| `GOOGLE_CLOUD_PROJECT` | Your GCP project ID | No (auto-detected) |
| `ENVIRONMENT` | `development` or `production` | No |

### Cloud Run Settings

- **Memory**: 2Gi (for FAISS vector database)
- **CPU**: 1 vCPU
- **Timeout**: 300 seconds
- **Auto-scaling**: 0-10 instances
- **Region**: us-central1

## 🎯 Post-Deployment

### Test Your API

```bash
# Get session ID
curl https://your-app-url.run.app/

# Send a message
curl -X POST https://your-app-url.run.app/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, I need help with stress management",
    "session_id": "your-session-id"
  }'
```

### Monitor Logs

```bash
# View logs
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=lucillellm" --limit=50

# Follow logs in real-time
gcloud logs tail "resource.type=cloud_run_revision AND resource.labels.service_name=lucillellm"
```

## 💰 Cost Estimation

- **Cloud Run**: ~$0.40/million requests
- **OpenAI GPT-4o-mini**: ~$0.15/1M input tokens, ~$0.60/1M output tokens
- **FAISS**: Included in memory allocation
- **Secrets**: Free tier available

## 🔒 Security

- API keys stored in Google Secret Manager
- HTTPS enforced by Cloud Run
- No authentication on endpoints (add if needed)
- CORS configured for swagger only

## 🐛 Troubleshooting

### Common Issues

1. **"No OpenAI API key found"**
   - Check your .env file
   - Verify secret exists: `gcloud secrets list`

2. **Memory issues**
   - FAISS database requires ~1.5Gi RAM
   - Increase memory if needed

3. **Cold starts**
   - First request after idle may take 10-30 seconds
   - Consider min-instances=1 for production

### Logs and Debugging

```bash
# Check build logs
gcloud builds log --region=global BUILD_ID

# Check service status
gcloud run services describe lucillellm --region=us-central1
```

## 🔄 Updates

To update your deployment:

```bash
# Make your code changes, then redeploy
gcloud builds submit --config cloudbuild.yaml
```

---

🎉 **Your LucilleLLM self-care chatbot is now live on Google Cloud!**
