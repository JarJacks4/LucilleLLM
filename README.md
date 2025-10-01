# LucilleLLM - Self-Care AI Assistant 🌟

A sophisticated conversational AI assistant specializing in self-care and mental wellness support. Built with FastAPI, OpenAI GPT-4, FAISS vector search, and Firebase for enterprise-grade scalability.

## ✨ Features

- 🤖 **AI-Powered Self-Care Assistant**: Lucille provides personalized, evidence-based self-care advice
- 💾 **Persistent Chat Storage**: All conversations securely stored in Firebase Firestore
- 🔄 **Intelligent Session Management**: Maintains conversation context with automatic summarization
- 🔍 **Knowledge Retrieval**: FAISS vector database for relevant self-care information lookup
- 📊 **Smart Conversation Summarization**: Automatically summarizes long conversations to maintain context
- 🎭 **Emotion Recognition API**: Separate microservice for facial emotion detection
- ☁️ **Cloud-Native**: Deployed on Google Cloud Run with automated CI/CD pipeline
- 🛡️ **Production-Ready**: Comprehensive error handling, rate limiting, and health monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   LucilleLLM     │    │   OpenAI API    │
│   (Web/Mobile)  │◄──►│   FastAPI        │◄──►│   GPT-4o-mini   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Firebase       │    │   FAISS Vector  │
                       │   Firestore      │    │   Database      │
                       └──────────────────┘    └─────────────────┘
```

## 📋 API Endpoints

### Core Chat API
- `POST /chat` - Send message and receive AI response
- `GET /chat-interface` - Simple web interface for testing
- `GET /health` - Health check with system status

### Session Management
- `GET /chat/{session_id}` - Retrieve chat history for a session
- `DELETE /chat/{session_id}` - Delete specific chat session
- `GET /sessions/` - List recent chat sessions

### System
- `GET /` - Root endpoint with session ID generation

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Cloud Project
- OpenAI API Key
- Firebase Project

### Local Development

1. **Clone and Install**
   ```bash
   git clone <repository-url>
   cd LucilleLLM
   pip install -r requirements.txt
   ```

2. **Environment Setup**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your credentials
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   ```

3. **Firebase Configuration**
   - Place your Firebase service account key in the project root
   - Update the filename in `firebase_service.py` if different
   - See [FIREBASE_SETUP.md](FIREBASE_SETUP.md) for detailed setup

4. **Start Development Server**
   ```bash
   python start_local.py
   # or
   python main.py
   ```

5. **Test the API**
   ```bash
   curl -X POST "http://localhost:8080/chat" \
        -H "Content-Type: application/json" \
        -d '{"message": "Hello Lucille!", "session_id": "test-session"}'
   ```

## 🚢 Deployment

### Google Cloud Run

1. **Build and Deploy**
   ```bash
   # Using Cloud Build
   gcloud builds submit --config cloudbuild.yaml
   
   # Or use the deployment script
   ./deploy.sh
   ```

2. **Environment Variables** (Set in Cloud Run)
   ```
   OPENAI_API_KEY=<stored-in-secret-manager>
   GOOGLE_CLOUD_PROJECT=<your-project-id>
   ```

### Deployment Architecture
- **Container**: Docker with Python 3.12 slim
- **Scaling**: Auto-scaling 0-10 instances
- **Security**: IAM, VPC, and secret management
- **Monitoring**: Cloud Logging and Error Reporting

## 📊 Database Schema

### Firebase Firestore Structure
```javascript
chat_sessions/{session_id}/
├── session_id: string
├── messages: array[
│   ├── type: "human" | "ai" | "system"
│   ├── content: string
│   └── timestamp: string
├── summary: string (auto-generated for long conversations)
├── created_at: timestamp
├── updated_at: timestamp
└── message_count: number
```

## 🔧 Configuration

### Key Configuration Files
- `main.py` - Main FastAPI application
- `firebase_service.py` - Firebase integration layer  
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `cloudbuild.yaml` - CI/CD pipeline
- `app.yaml` - App Engine configuration (alternative)

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | Yes |
| `FAISS_INDEX_PATH` | Path to FAISS vector database | No (default: ./faiss_vecdb) |

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8080/health
```

### Chat Test
```bash
curl -X POST "http://localhost:8080/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "I'\''m feeling stressed about work. Can you help?",
       "session_id": "test-session-123"
     }'
```

## 🛠️ Development

### Project Structure
```
LucilleLLM/
├── main.py                 # Main FastAPI application
├── firebase_service.py     # Firebase integration
├── chat_service.py         # Core chat logic
├── chat_agent_service.py   # LangChain agent setup
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── cloudbuild.yaml        # CI/CD pipeline
├── faiss_vecdb/          # Vector database files
├── vit_emotion_api/      # Emotion recognition microservice
└── docs/                 # Documentation
    ├── DEPLOYMENT.md
    └── FIREBASE_SETUP.md
```

### Adding Features
1. **New Endpoints**: Add to `main.py` with proper error handling
2. **Chat Logic**: Extend `chat_service.py` for new conversation features
3. **Database**: Update `firebase_service.py` for new data operations
4. **Deployment**: Update `cloudbuild.yaml` for new build steps

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🆘 Support

- **Documentation**: See `/docs` folder for detailed guides
- **Issues**: Create an issue in the repository
- **Health Check**: Use `/health` endpoint to verify system status

---

**Built with ❤️ for mental wellness and self-care support**