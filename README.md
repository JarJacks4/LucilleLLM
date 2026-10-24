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

<<<<<<< HEAD
## 🛠️ Development
=======
```json
{
  "session_id": "demo-1",
  "response": "I hear you — work stress can feel overwhelming. Let's explore what's weighing on you most...",
  "detected_emotion": "anxious",
  "detected_intent": "venting",
  "status": "success"
}
```

```bash
# Start a WHO-5 well-being assessment
curl -X POST http://localhost:8080/assessments/demo-user/start \
  -H "Content-Type: application/json" \
  -d '{"assessment_type": "who5"}'
```

```bash
# Get wellness score
curl http://localhost:8080/assessments/demo-user/wellness-score
```

```json
{
  "overall_score": 72,
  "overall_label": "Moderate Well-being",
  "phq9_score": 7,
  "phq9_severity": "Mild Depression",
  "gad7_score": 4,
  "gad7_severity": "Minimal Anxiety",
  "disclaimer": "This score is from a standardized screening tool and is not a clinical diagnosis."
}
```

---

## Escape v1 API

The Escape app's Journal, Mood Scan / Mood Stats, Soundscapes AI, Self-Care Score + plans and privacy controls run on
`/v1` endpoints in `escape_api/` (67 operations, Firebase-token auth, consent-gated, cost-capped). Legacy routes are unchanged.
See [API_REFERENCE.md](API_REFERENCE.md) sections 20–27, [docs/API_V1.md](docs/API_V1.md) and
[docs/escape_v1_openapi.json](docs/escape_v1_openapi.json) (import into FlutterFlow).

## Core Features

### Therapy Engine
4 evidence-based modalities with guided multi-step exercises, practice tasks, progress tracking, and ambient soundscapes.

| Modality | Focus | Exercises |
|----------|-------|-----------|
| **CBT** | Thought restructuring | Thought records, distortion spotting, behavioral activation |
| **ACT** | Acceptance & values | Values exploration, mindfulness, acceptance exercises |
| **DBT** | Emotion regulation | Distress tolerance, emotion regulation, interpersonal skills |
| **MI** | Motivation & change | Ambivalence exploration, value clarification, scaling |

### Mental Health Assessments
Clinically validated, public domain instruments with deterministic scoring (no AI interpretation).

| Instrument | Measures | Score | Threshold |
|------------|----------|-------|-----------|
| **PHQ-9** | Depression | 0-27 | >= 10 flagged |
| **GAD-7** | Anxiety | 0-21 | >= 10 flagged |
| **WHO-5** | Well-being | 0-100 | < 50 flagged |

PHQ-9 Question 9 (self-harm ideation) triggers immediate crisis protocol regardless of total score.

### 5-Layer Personality

| Layer | What It Tracks |
|-------|---------------|
| Persona | Communication style, personality traits, interests |
| Affective | Mood history, emotional triggers, current state |
| Behavioral | Sleep patterns, exercise habits, daily routines |
| Motivational | Goals, core values, motivations |
| Cognitive | Beliefs, thought patterns, cognitive distortions |

### Safety System
- **Crisis detection** — Keyword matching at 3 severity levels (bypasses LLM entirely)
- **Jailbreak blocking** — Prompt injection pattern detection
- **Output validation** — Screens AI responses before delivery
- **Dependency monitoring** — 7-signal usage pattern analysis prevents unhealthy reliance
- **Auto-escalation** — Creates human review tickets for critical events

### Compliance
- **GDPR Article 17** — Cascade deletion across 23 Firestore collections
- **GDPR Article 20** — Full data portability export
- **HIPAA-aligned** — 7-year audit trail with structured logging
- **Consent management** — Per-purpose consent tracking

---

## API Overview

75+ endpoints organized by domain. Full interactive docs at `/docs` when running.

| Domain | Endpoints | Key Routes |
|--------|-----------|------------|
| Chat | 6 | `POST /chat`, `POST /chat/stream`, `POST /chat/voice` |
| Users | 16 | `POST /users/onboard`, `GET /users/{id}`, `DELETE /users/{id}/data` |
| Therapy | 13 | `POST /therapy/{id}/start`, `GET /therapy/{id}/progress` |
| Assessments | 8 | `POST /assessments/{id}/start`, `GET /assessments/{id}/wellness-score` |
| Feedback | 4 | `POST /feedback/{id}/exercise-outcome` |
| Soundscapes | 8 | `GET /soundscapes/recommend/{id}` |
| Safety | 3 | `GET /safety/resources`, `POST /safety/check` |
| Voice | 2 | `POST /tts`, `POST /stt` |
| Wearables | 4 | `POST /wearables/{id}/sync`, `GET /wearables/{id}/sleep-insights` |
| Admin | 16 | `GET /admin/dashboard`, `GET /admin/escalations` (auth required) |
| Fine-Tuning | 6 | `POST /finetuning/submit-job`, `GET /finetuning/ab-stats` (auth required) |
| Reviews | 3 | `POST /reviews/{id}/generate` |
| Health | 4 | `GET /health`, `GET /health/detailed` |

See [API_REFERENCE.md](API_REFERENCE.md) for complete endpoint documentation.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI + Pydantic v2 |
| LLM | OpenAI GPT-4o-mini |
| RAG | FAISS + text-embedding-3-small |
| Database | Firebase Firestore (23 collections) |
| Auth | Firebase Auth JWT + API key |
| Assessments | PHQ-9, GAD-7, WHO-5 (public domain) |
| Voice | edge-tts + SpeechRecognition |
| RL | Thompson Sampling (bandit) |
| Retry | tenacity (exponential backoff) |
| Testing | pytest (78 tests) |
| Deploy | Docker + Google Cloud Run + Cloud Build |

---

## Project Structure
>>>>>>> cbc7d48bf869c367e4478971c41d52a25c319bc2

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

<<<<<<< HEAD
**Built with ❤️ for mental wellness and self-care support**
=======
## Authentication

| Method | For | How |
|--------|-----|-----|
| Firebase JWT | Mobile/web clients | `Authorization: Bearer <firebase_id_token>` |
| Internal Service Key | Service-to-service | `Authorization: Bearer <INTERNAL_SERVICE_KEY>` |
| None | Public endpoints | `/health`, `/`, `/assessments/instruments` |

Admin endpoints (`/admin/*`, `/finetuning/*`) require admin role.

---

## Testing

```bash
pip install pytest pytest-asyncio httpx
python -m pytest tests/ -v
```

**78 tests** covering: crisis detection, dependency scoring, PHQ-9/GAD-7/WHO-5 scoring algorithms, input sanitization, model validation, and config defaults.

Tests run automatically in Cloud Build before every deployment.

---

## Deployment

```bash
# Cloud Build (builds, tests, deploys)
gcloud builds submit --config cloudbuild.yaml

# Docker
docker build -t lucille-llm .
docker run -p 8080:8080 --env-file .env lucille-llm
```

**Cloud Run config**: 2 CPU, 4GB RAM, 0-10 instances, 600s timeout, secrets via Secret Manager.

---

## Configuration

All config via environment variables. See [.env.example](.env.example) for the complete list.

| Required | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID |

| Group | Key Variables | Defaults |
|-------|-------------|----------|
| Model | `OPENAI_MODEL` | gpt-4o-mini |
| Rate Limiting | `RATE_LIMIT_CHAT` | 10/min |
| Safety | `ESCALATION_ENABLED` | true |
| Assessments | `ASSESSMENT_ENABLED`, thresholds | true, PHQ-9/GAD-7 >= 10, WHO-5 < 50 |
| RL | `RL_ENABLED` | true |
| Fine-Tuning | `FT_ENABLED` | false |
| Data Retention | 8 retention configs | 180 days to 7 years |

---

## Development Phases

| Phase | Name | Key Additions |
|-------|------|---------------|
| 1-3 | Core Chat | FastAPI, OpenAI, streaming SSE, FAISS RAG |
| 4-5 | User System | 5-layer profiles, onboarding, Firebase |
| 6 | Emotion & Memory | Emotion detection, intent, memory system |
| 7 | Therapy Engine | CBT/ACT/DBT/MI exercises, progress tracking |
| 8 | Soundscapes | Ambient audio, exercise-soundscape mapping |
| 9 | Safety | Crisis detection, jailbreak, output validation |
| 10 | Feedback | Response ratings, exercise outcomes |
| 11 | Prompt Engine | 5-layer dynamic prompt assembly |
| 12 | Dependency | 7-signal usage monitoring, cooldown |
| 13 | Cultural | Country-aware recommendations, bias checking |
| 14 | Compliance | GDPR, consent, HIPAA audit logging |
| 15 | Voice | TTS, STT, GCS audio storage |
| 16 | Wearables | Health data sync, sleep insights |
| 17 | RL | Thompson Sampling bandits |
| 18 | Fine-Tuning | Training extraction, A/B testing |
| 19 | Dashboard | Admin dashboard, 8 metric endpoints |
| 20 | Escalation | Auto-escalation queue, annual reviews |
| 21 | Assessments & Hardening | PHQ-9/GAD-7/WHO-5, auth, sanitization, tests, CI |

---

## Firestore Schema

23 collections across user data, therapy, safety, compliance, and ML:

```
user_profiles/{user_id}                  # 5-layer user profiles
chat_sessions/{session_id}               # Chat messages & summaries
user_memories/{user_id}/memories/*       # Episodic/semantic/factual
exercise_sessions/{user_id}/sessions/*   # Therapy sessions & tasks
assessments/{user_id}/sessions/*         # PHQ-9/GAD-7/WHO-5 results
feedback/{user_id}/*                     # Response + exercise feedback
safety_audit/{user_id}/events/*          # Crisis/safety events
health_metrics/{user_id}/daily/*         # Wearable data
bandit_state/{user_id}                   # RL state
escalation_queue/{escalation_id}         # Human review tickets
audit_log/{entry_id}                     # HIPAA trail (7yr)
consent_records/{user_id}                # GDPR consent
+ 11 more collections
```

---

## Status

**Current: Phase 21** — Production hardening complete. Auth, tests, assessments, input sanitization, retry logic, and CI pipeline all in place.

**Next up:**
- Split main.py into APIRouters (infrastructure ready in `routes/`)
- Redis-backed distributed rate limiting
- External FAISS index loading from GCS

See [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) for the full checklist.

---

## Contributing

1. Fork and create a feature branch
2. Follow the singleton service pattern (`*_service.py` + `get_*_service()`)
3. Add models to `models.py`, config to `config.py`
4. Protect sensitive endpoints with auth middleware
5. Sanitize user inputs via `utils/sanitize.py`
6. Write tests in `tests/` — all 78 must pass
7. Update `.env.example` and GDPR compliance if adding user data
8. Open a PR

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for architecture patterns and conventions.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Related Docs

| Document | Description |
|----------|-------------|
| [API_REFERENCE.md](API_REFERENCE.md) | Complete endpoint documentation (75+ endpoints) |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Architecture patterns, service conventions |
| [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) | Production hardening checklist |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment instructions |
| [FIREBASE_SETUP.md](FIREBASE_SETUP.md) | Firestore configuration |
>>>>>>> cbc7d48bf869c367e4478971c41d52a25c319bc2
