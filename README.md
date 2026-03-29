# LucilleLLM

**An AI therapeutic companion that understands you better over time.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-78%20passing-brightgreen.svg)](tests/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com)
[![Cloud Run](https://img.shields.io/badge/Deploy-Cloud%20Run-4285F4.svg)](https://cloud.google.com/run)

LucilleLLM is a production-grade mental wellness API that combines therapy techniques (CBT, ACT, DBT, MI), clinically validated assessments (PHQ-9, GAD-7, WHO-5), and a 5-layer personality engine to deliver personalized, evidence-based self-care support.

Built across 21 development phases. Deployed on Google Cloud Run.

---

## Why LucilleLLM?

Most mental health chatbots give generic responses. LucilleLLM is different:

- **Remembers you** — Episodic, semantic, and factual memory that persists across sessions
- **Adapts to you** — 5-layer personality model learns your communication style, goals, and triggers
- **Clinically grounded** — PHQ-9, GAD-7, WHO-5 assessments use published scoring algorithms (no AI guessing)
- **Safety-first** — Crisis detection bypasses the LLM entirely and returns helplines immediately
- **Learns what works** — Thompson Sampling RL optimizes which therapy techniques help you most
- **Compliant** — GDPR export/deletion, HIPAA audit logging, 7-year retention trails

---

## How It Works

```
Client Request
    |
    v
Input Sanitization --> Auth (JWT/API Key)
    |
    v
Emotion Detection --> Safety Screening --> Dependency Check
    |
    v
RAG (FAISS) --> Memory Retrieval --> RL Modality Selection
    |
    v
5-Layer Prompt Assembly --> OpenAI GPT-4o-mini (with retry)
    |
    v
Output Validation --> Auto-Escalation Check
    |
    v
Response + Session Storage (Firestore)
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key
- Google Cloud project with Firestore

### Setup

```bash
git clone <repository-url>
cd LucilleLLM
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set OPENAI_API_KEY and GOOGLE_CLOUD_PROJECT

python start_local.py
# API at http://localhost:8080
# Docs at http://localhost:8080/docs
```

### Try It

```bash
# Chat
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I feel stressed about work", "session_id": "demo-1"}'
```

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

```
LucilleLLM/
  main.py                  # FastAPI app, 75+ endpoints
  models.py                # 85+ Pydantic models
  config.py                # 73 config fields (env vars)
  auth_middleware.py        # JWT + API key auth

  *_service.py (20 files)  # Singleton services:
    assessment_, therapy_, safety_, dependency_, compliance_,
    emotion_, memory_, rl_, finetuning_, escalation_, ...

  utils/                   # sanitize.py, retry.py, logging.py
  tests/                   # 78 tests across 6 files
  routes/                  # Router migration (in progress)

  Dockerfile               # Python 3.12-slim
  cloudbuild.yaml           # CI/CD with test step
```

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for detailed architecture and file-by-file documentation.

---

## Authentication

| Method | For | How |
|--------|-----|-----|
| Firebase JWT | Mobile/web clients | `Authorization: Bearer <firebase_id_token>` |
| API Key | Service-to-service | `Authorization: Bearer <LUCILLE_API_KEY>` |
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
