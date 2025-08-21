# LucilleLLM - Self-Care AI Assistant

A conversational AI assistant specializing in self-care and mental wellness support, built with FastAPI, OpenAI, and Firebase.

## Features

- 🤖 **AI-Powered Self-Care Assistant**: Lucille provides personalized self-care advice and support
- 💾 **Persistent Chat Storage**: All conversations are stored in Firebase Firestore
- 🔄 **Session Management**: Maintains conversation context across sessions
- 📊 **Conversation Summarization**: Automatically summarizes long conversations
- 🎭 **Emotion Recognition**: Separate API for facial emotion detection
- ☁️ **Cloud-Ready**: Deployed on Google Cloud Run with automated CI/CD

## Firebase Integration

The application now uses Firebase Firestore for persistent chat storage:

### Database Schema
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

### API Endpoints

- `POST /chat` - Send message and get response
- `GET /chat/{session_id}` - Retrieve chat history
- `DELETE /chat/{session_id}` - Delete chat session
- `GET /sessions/` - List recent sessions
- `GET /health` - Health check with Firebase connectivity

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Firebase Configuration**
   - **Quick Setup**: Run `python firebase_diagnostic.py` to check your setup
   - **Detailed Setup**: See [FIREBASE_SETUP.md](FIREBASE_SETUP.md) for complete instructions
   - Firebase project: `escape-ujuzxr`

3. **Environment Variables**
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_CLOUD_PROJECT=escape-ujuzxr
   ```

4. **Run Locally**
   ```bash
   python main.py
   ```

5. **Test Firebase Integration**
   ```bash
   python test_firebase.py
   python test_main_integration.py
   ```

## Deployment

The application is configured for Google Cloud Run deployment:

- **Container**: Docker with Python 3.12
- **CI/CD**: Cloud Build with automated deployment
- **Secrets**: Google Secret Manager for API keys
- **Scaling**: Auto-scaling with 0-10 instances

## Architecture

- **Backend**: FastAPI with async support
- **AI**: OpenAI GPT-4o-mini + embeddings
- **Vector DB**: FAISS for knowledge retrieval
- **Storage**: Firebase Firestore for chat persistence
- **Deployment**: Google Cloud Run