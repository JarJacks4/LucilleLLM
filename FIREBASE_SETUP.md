# Firebase Setup Guide for LucilleLLM

This guide will help you set up Firebase Firestore for persistent chat storage in LucilleLLM.

## 🔧 Prerequisites

1. **Google Cloud Project**: You need a Google Cloud project with billing enabled
2. **Firebase Project**: A Firebase project linked to your Google Cloud project
3. **Google Cloud CLI**: For authentication (optional but recommended)

## 📋 Step-by-Step Setup

### 1. Enable Firebase Firestore

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project (`escape-ujuzxr`)
3. In the left sidebar, click **Firestore Database**
4. Click **Create Database**
5. Choose **Start in test mode** (for development) or **Start in production mode** (for production)
6. Select a location (choose the same region as your Cloud Run deployment)
7. Click **Done**

### 2. Set Up Authentication

#### Option A: Google Cloud Default Credentials (Recommended for Production)

```bash
# Install Google Cloud CLI if not already installed
# https://cloud.google.com/sdk/docs/install

# Authenticate with Google Cloud
gcloud auth login

# Set your project
gcloud config set project escape-ujuzxr

# Set up application default credentials
gcloud auth application-default login
```

#### Option B: Service Account Key (Alternative)

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project
3. Go to **Project Settings** > **Service Accounts**
4. Click **Generate New Private Key**
5. Download the JSON file
6. Place it in your project root as `escape-ujuzxr-firebase-adminsdk-he895-1a039bd95a.json`

### 3. Set Environment Variables

```bash
# For local development
export GOOGLE_CLOUD_PROJECT=escape-ujuzxr
export FIREBASE_PROJECT_ID=escape-ujuzxr

# For Cloud Run deployment, these are already set in cloudbuild.yaml
```

### 4. Test the Setup

Run the diagnostic script to verify everything is working:

```bash
# Activate your virtual environment
source venv/bin/activate

# Run diagnostic
python firebase_diagnostic.py

# Test Firebase integration
python test_firebase.py

# Test main application
python test_main_integration.py
```

## 🚀 Deployment Configuration

### For Cloud Run

The application is already configured to use Google Cloud's default credentials in production. The `cloudbuild.yaml` file includes:

```yaml
- '--set-env-vars=ENVIRONMENT=production'
- '--set-secrets=OPENAI_API_KEY=openai-api-key:latest'
```

### For Local Development

Create a `.env` file in your project root:

```env
GOOGLE_CLOUD_PROJECT=escape-ujuzxr
FIREBASE_PROJECT_ID=escape-ujuzxr
OPENAI_API_KEY=your_openai_api_key
```

## 🔍 Troubleshooting

### Common Issues

1. **"Invalid JWT Signature" Error**
   - Your service account key has expired
   - Solution: Use Google Cloud default credentials instead

2. **"Database does not exist" Error**
   - Firestore database is not enabled
   - Solution: Enable Firestore in Firebase Console

3. **"Permission denied" Error**
   - Service account lacks necessary permissions
   - Solution: Ensure the service account has Firestore permissions

### Diagnostic Commands

```bash
# Check Firebase connection
python firebase_diagnostic.py

# Test Firebase operations
python test_firebase.py

# Test main application
python test_main_integration.py

# Check Google Cloud authentication
gcloud auth list
gcloud config list
```

## 📊 Database Schema

The application creates the following structure in Firestore:

```
chat_sessions/
├── {session_id}/
    ├── session_id: string
    ├── messages: array
    │   ├── type: "human" | "ai" | "system"
    │   ├── content: string
    │   └── timestamp: string
    ├── summary: string
    ├── created_at: timestamp
    ├── updated_at: timestamp
    └── message_count: number
```

## 🔒 Security Rules

For production, set up Firestore security rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /chat_sessions/{sessionId} {
      allow read, write: if request.auth != null;
    }
  }
}
```

## 📈 Monitoring

Monitor your Firebase usage in the [Firebase Console](https://console.firebase.google.com/):

- **Firestore**: Database usage and performance
- **Authentication**: User authentication (if implemented)
- **Analytics**: Usage analytics (optional)

## ✅ Verification Checklist

- [ ] Firestore database is enabled
- [ ] Authentication is working (default credentials or service account)
- [ ] Diagnostic script passes all tests
- [ ] Firebase integration test passes
- [ ] Main application loads successfully
- [ ] Chat sessions are being stored and retrieved
- [ ] Environment variables are set correctly
- [ ] Cloud Run deployment is configured (for production)

## 🆘 Getting Help

If you encounter issues:

1. Check the diagnostic output
2. Verify your Firebase project settings
3. Ensure your Google Cloud project has billing enabled
4. Check that Firestore is enabled in the correct region
5. Verify your authentication method is working

For additional support, check the [Firebase Documentation](https://firebase.google.com/docs) or [Google Cloud Documentation](https://cloud.google.com/docs).

