# LucilleLLM Developer Guide

A comprehensive reference for understanding, maintaining, and extending the LucilleLLM codebase. This document covers architecture patterns, service conventions, data flow, and step-by-step guides for common modifications.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Service Layer Pattern](#2-service-layer-pattern)
3. [The /chat Pipeline (Detailed)](#3-the-chat-pipeline-detailed)
4. [Models & Data Contracts](#4-models--data-contracts)
5. [Configuration System](#5-configuration-system)
6. [Firestore Data Model](#6-firestore-data-model)
7. [Prompt Engine & Personality System](#7-prompt-engine--personality-system)
8. [Safety & Crisis Pipeline](#8-safety--crisis-pipeline)
9. [Reinforcement Learning System](#9-reinforcement-learning-system)
10. [Fine-Tuning Pipeline](#10-fine-tuning-pipeline)
11. [GDPR Compliance Layer](#11-gdpr-compliance-layer)
12. [Middleware Stack](#12-middleware-stack)
13. [Caching Strategy](#13-caching-strategy)
14. [File-by-File Reference](#14-file-by-file-reference)
15. [How To: Add a New Feature (Phase)](#15-how-to-add-a-new-feature-phase)
16. [How To: Add a New API Endpoint](#16-how-to-add-a-new-api-endpoint)
17. [How To: Add a New Firestore Collection](#17-how-to-add-a-new-firestore-collection)
18. [How To: Add a New Config Field](#18-how-to-add-a-new-config-field)
19. [How To: Integrate into the /chat Pipeline](#19-how-to-integrate-into-the-chat-pipeline)
20. [Common Pitfalls](#20-common-pitfalls)
21. [Testing Conventions](#21-testing-conventions)
22. [Deployment Notes](#22-deployment-notes)

---

## 1. System Architecture

### High-Level Data Flow

```
Client Request
      |
      v
+------------------+
|   Middleware       |  Rate Limiting -> Metrics -> Privacy
+------------------+
      |
      v
+------------------+
|   main.py          |  FastAPI route handler (~4,200 lines)
|   /chat endpoint   |
+------------------+
      |
      +---> emotion_service      (detect emotion + intent)
      +---> safety_service       (crisis/jailbreak screening)
      +---> dependency_service   (usage pattern check)
      +---> cultural_service     (cultural context extraction)
      +---> wearable_service     (health data context)
      +---> memory_service       (recall user memories)
      +---> rl_service           (Thompson Sampling recommendation)
      +---> prompt_engine        (build layered system prompt)
      +---> chat_agent_service   (LangChain agent w/ tools)
      +---> OpenAI API           (generate response)
      +---> safety_service       (output validation)
      +---> escalation_service   (auto-escalation check)
      +---> firebase_service     (persist session)
      +---> finetuning_service   (A/B routing + performance logging)
      |
      v
Client Response
```

### Singleton Service Architecture

Every service follows the singleton pattern:

```python
# In service_file.py
class MyService:
    def __init__(self):
        self._config = get_config()
        self._db = None
        try:
            from firebase_service import get_firebase_service
            fb = get_firebase_service()
            self._db = fb.db
        except Exception as e:
            logger.warning(f"MyService: Firebase not available -- {e}")

_my_service: Optional[MyService] = None

def get_my_service() -> MyService:
    global _my_service
    if _my_service is None:
        _my_service = MyService()
    return _my_service
```

**Why singletons?** Services hold Firebase connections, OpenAI clients, and cached state. Creating them once at startup avoids repeated initialization costs and ensures consistent state.

### Service Dependency Graph

```
firebase_service  <----+---- All services that need Firestore
                       |
config (AppConfig) <---+---- All services read configuration
                       |
openai_client     -----+--> emotion_service (injected via constructor)
                       +--> memory_service  (injected via constructor)
                       +--> finetuning_service (injected via set_openai_client)
                       +--> escalation_service (injected via set_openai_client)
                       +--> chat_agent_service (used internally)
```

Services that need the OpenAI client receive it via one of two patterns:
1. **Constructor injection**: `get_emotion_service(openai_client=client)` -- for services initialized before FastAPI app creation
2. **Setter injection**: `svc.set_openai_client(client)` -- for services that initialize during startup (after app creation)

---

## 2. Service Layer Pattern

### Standard Service Structure

Every service file follows this template:

```
"""
LucilleLLM - [Service Name] (Phase N)
One-line description.
"""

import logging
from typing import Optional, List
from config import get_config

logger = logging.getLogger(__name__)

class MyService:
    """Docstring with purpose and key behaviors."""

    COLLECTION_NAME = "my_collection"  # Firestore collection constant

    def __init__(self):
        self._config = get_config()
        self._db = None

        try:
            from firebase_service import get_firebase_service
            fb = get_firebase_service()
            self._db = fb.db
        except Exception as e:
            logger.warning(f"MyService: Firebase not available -- {e}")

        logger.info(f"MyService initialized -- db_available={self._db is not None}")

    @property
    def is_available(self) -> bool:
        return self._db is not None

    # ... business methods ...

# Singleton
_my_service: Optional[MyService] = None

def get_my_service() -> MyService:
    global _my_service
    if _my_service is None:
        _my_service = MyService()
    return _my_service
```

### Key Conventions

| Convention | Pattern | Why |
|-----------|---------|-----|
| Graceful degradation | All methods return defaults (None, [], {}) when DB unavailable | App runs without Firebase for local dev |
| Try/except everywhere | Every DB operation wrapped | One failed query never crashes the app |
| Lazy imports | `from firebase_service import ...` inside `__init__` | Avoids circular import issues |
| Collection constants | `COLLECTION_NAME = "..."` as class variable | Single source of truth for collection names |
| Logging | `logger = logging.getLogger(__name__)` | Structured JSON logging in production |

### Service Inventory

| Service | File | Singleton Getter | Phase |
|---------|------|-------------------|-------|
| FirebaseService | firebase_service.py | `get_firebase_service()` | 1 |
| UserService | user_service.py | `get_user_service()` | 2 |
| EmotionService | emotion_service.py | `get_emotion_service()` | 3 |
| MemoryService | memory_service.py | `get_memory_service()` | 4 |
| TherapyService | therapy_service.py | `get_therapy_service()` | 5 |
| ProgressService | progress_service.py | `get_progress_service()` | 6 |
| FeedbackService | feedback_service.py | `get_feedback_service()` | 7 |
| SoundscapeService | soundscape_service.py | `get_soundscape_service()` | 8 |
| SafetyService | safety_service.py | `get_safety_service()` | 9 |
| PromptEngine | prompt_engine.py | `get_prompt_engine()` | 11 |
| DependencyService | dependency_service.py | `get_dependency_service()` | 12 |
| CulturalService | cultural_service.py | `get_cultural_service()` | 13 |
| ComplianceService | compliance_service.py | `get_compliance_service()` | 14 |
| AuditService | audit_service.py | `get_audit_service()` | 14 |
| StorageService | storage_service.py | `get_storage_service()` | 15 |
| VoiceService | voice_service.py | `get_voice_service()` | 15 |
| WearableService | wearable_service.py | `get_wearable_service()` | 16 |
| RLService | rl_service.py | `get_rl_service()` | 17 |
| FineTuningService | finetuning_service.py | `get_finetuning_service()` | 18 |
| MonitoringService | monitoring_service.py | `get_monitoring_service()` | 19 |
| EscalationService | escalation_service.py | `get_escalation_service()` | 20 |
| ChatAgentService | chat_agent_service.py | `get_chat_agent_service()` | 1 |
| TTLCache | cache.py | `get_cache()` | 10 |

---

## 3. The /chat Pipeline (Detailed)

The `/chat` endpoint in `main.py` is the most complex route (~300 lines). Here is the exact order of operations:

```
POST /chat (ChatRequest) -> ChatResponse
|
|-- 1. Session resolution (existing or new session_id)
|-- 2. Rate limiting check
|-- 3. Load user profile (if user_id provided)
|-- 4. Emotion detection (OpenAI call)
|      -> EmotionDetectionResult {emotion, intent, confidence}
|-- 5. Phase 9: Input safety screening (keyword-based, NO API call)
|      -> SafetyCheckResult {risk_level, flags, crisis_detected, jailbreak_detected}
|
|-- 6. CRITICAL CHECK: If risk_level == CRITICAL
|      |-- Return crisis helpline response immediately (bypass LLM)
|      |-- Log safety event
|      |-- Phase 20: Auto-escalation (URGENT priority)
|      |-- RETURN early
|
|-- 7. Phase 12: Dependency detection (fast, no API call)
|      -> DependencyAssessment {risk_level, score, signals}
|      -> dependency_override_text injected into prompt if triggered
|
|-- 8. Phase 13: Cultural context extraction
|      -> CulturalContext {country, region, ...}
|
|-- 9. Retrieve RAG context from FAISS vector store
|      -> Top-5 relevant self-care knowledge chunks
|
|-- 10. Load user memories (semantic search)
|-- 11. Load conversation summary (if long session)
|-- 12. Load active exercise session context
|-- 13. Load due practice tasks
|
|-- 14. Phase 7/17: Effectiveness + RL recommendation
|       -> Therapy modality recommendation with RL bonus
|
|-- 15. Phase 8: Soundscape suggestion lookup
|-- 16. Phase 16: Wearable health context
|
|-- 17. Build system prompt via prompt_engine
|       (persona + memories + profile + summary + all overlays)
|
|-- 18. Phase 18: A/B model routing (base vs fine-tuned)
|
|-- 19. OpenAI API call (with system prompt + chat history)
|       -> bot_response
|
|-- 20. Phase 9: Output safety validation
|       -> Potentially modify response if unsafe content detected
|
|-- 21. Log safety event (if risk_level != LOW)
|       -> Capture safety_event_id
|
|-- 22. Phase 20: Auto-escalation check
|       -> Creates ticket if CRITICAL/HIGH safety or HIGH dependency
|
|-- 23. Update chat history
|-- 24. Persist session to Firebase
|-- 25. Phase 13: Training data collection (if consent given)
|-- 26. Phase 18: Log model performance (for A/B comparison)
|
|-- 27. Return ChatResponse
```

The `/chat/stream` endpoint follows the same pipeline but uses SSE (Server-Sent Events) for token-by-token streaming. It has identical safety/escalation integration points.

### Critical Integration Points

When adding new features to the chat pipeline, these are the key insertion points (search for comments in main.py):

| Location | After | Purpose |
|----------|-------|---------|
| Pre-LLM | Safety screening | Add new pre-processing checks |
| Pre-LLM | Dependency detection | Add context that modifies the prompt |
| Pre-LLM | Prompt building | Add new context overlays |
| Post-LLM | Output validation | Add post-processing on response |
| Post-LLM | Safety event log | Add new logging/tracking |
| Post-LLM | Session persistence | Add new data capture |

---

## 4. Models & Data Contracts

All Pydantic models live in `models.py` (~1,250 lines, 70+ models).

### Model Categories

**Enums (12):**
- `CommunicationStyle` - empathetic, direct, analytical, casual
- `DetectedEmotion` - sad, anxious, angry, happy, neutral, etc.
- `UserIntent` - crisis, seeking_advice, venting, reflecting, etc.
- `TherapyModality` - cbt, act, dbt, mi
- `MemoryType` - episodic, semantic, factual
- `TaskStatus` - pending, completed, skipped, overdue
- `FeedbackRating` - helpful, somewhat_helpful, not_helpful, harmful
- `RiskLevel` - LOW, MEDIUM, HIGH, CRITICAL
- `SafetyEventType` - CRISIS_DETECTED, HIGH_RISK_INPUT, JAILBREAK_ATTEMPT, etc.
- `DependencyRiskLevel` - NONE, LOW, MODERATE, HIGH
- `SoundscapeCategory` - nature, ambient, music, guided
- `EscalationTriggerType` / `EscalationPriority` / `EscalationStatus`

**Request/Response Models:** Follow the pattern `XxxRequest` / `XxxResponse` for API contracts.

**Internal Models:** Used for service-to-service data transfer (e.g., `SafetyCheckResult`, `DependencyAssessment`, `EffectivenessProfile`).

### Adding a New Model

1. Add it to `models.py` in the appropriate section
2. Use `Field(default_factory=...)` for mutable defaults (lists, dicts)
3. Use `model_dump()` for serialization (Pydantic v2)
4. Import it in `main.py` inside the models import block

---

## 5. Configuration System

### How It Works

```python
# config.py
@dataclass(frozen=True)           # Immutable after creation
class AppConfig:
    MY_FIELD: int = 42             # Type-annotated with default

def _load_config() -> AppConfig:   # Reads os.environ
    return AppConfig(
        MY_FIELD=_env_int("MY_FIELD", 42),
    )

_config: Optional[AppConfig] = None

def get_config() -> AppConfig:     # Singleton
    global _config
    if _config is None:
        _config = _load_config()
    return _config
```

### Helper Functions

| Function | Purpose |
|----------|---------|
| `_env_int(key, default)` | Read env var as int, fallback to default |
| `_env_float(key, default)` | Read env var as float |
| `_env_bool(key, default)` | Read env var as bool (`true/1/yes` = True) |
| `os.getenv(key, default)` | Read env var as string |

### Configuration Sections (68 fields total)

| Section | Count | Examples |
|---------|-------|----------|
| Environment | 1 | ENVIRONMENT |
| Model | 3 | OPENAI_MODEL, EMBEDDING_MODEL, FAISS_INDEX_PATH |
| Rate Limiting | 3 | RATE_LIMIT_CHAT, RATE_LIMIT_GLOBAL |
| Cache TTLs | 4 | CACHE_TTL_USER_PROFILE, CACHE_MAX_SIZE |
| Logging | 2 | LOG_LEVEL, LOG_FORMAT |
| Metrics | 1 | METRICS_ENABLED |
| Dependency | 8 | DEPENDENCY_MAX_MESSAGES_PER_HOUR, etc. |
| Cultural | 2 | CULTURAL_BIAS_CHECK_ENABLED, CULTURAL_DEFAULT_COUNTRY |
| Data Retention | 8 | RETENTION_CHAT_SESSIONS_DAYS, etc. |
| Audit | 1 | AUDIT_LOG_ENABLED |
| Audio/GCS | 3 | GCS_AUDIO_BUCKET, etc. |
| Voice I/O | 5 | TTS_PROVIDER, STT_PROVIDER, etc. |
| Wearables | 3 | WEARABLE_SYNC_ENABLED, etc. |
| RL | 5 | RL_ENABLED, RL_EXPLORATION_BONUS, etc. |
| Fine-Tuning | 9 | FT_ENABLED, FT_MODEL_ID, FT_AB_SPLIT_PERCENT, etc. |
| Dashboard | 3 | DASHBOARD_ENABLED, DASHBOARD_CACHE_TTL, etc. |
| Escalation | 3 | ESCALATION_ENABLED, thresholds |
| Reviews | 2 | REVIEW_ENABLED, REVIEW_DEFAULT_PERIOD_DAYS |

---

## 6. Firestore Data Model

### Collection Patterns

**Pattern 1: Single Document Per User**
```
collection/{user_id}  ->  { field: value, ... }
```
Used by: `user_profiles`, `bandit_state`, `interaction_metrics`, `consent_records`

**Pattern 2: Subcollection Per User**
```
collection/{user_id}/subcollection/{doc_id}  ->  { field: value, ... }
```
Used by: `user_memories/*/memories`, `feedback/*/response_feedback`, `safety_audit/*/events`, `health_metrics/*/daily`, `annual_reviews/*/reviews`, etc.

**Pattern 3: Flat Collection (Query by user_id)**
```
collection/{doc_id}  ->  { user_id: "...", ... }
```
Used by: `chat_sessions`, `training_examples`, `model_performance`, `escalation_queue`, `finetuning_jobs`

### Collection Reference

| Collection | Pattern | Key Field | Retention |
|-----------|---------|-----------|-----------|
| user_profiles | Single doc | user_id | Until deletion |
| chat_sessions | Flat, query | session_id + user_id | 365 days |
| user_memories/{uid}/memories | Subcollection | memory_id | 730 days |
| feedback/{uid}/response_feedback | Subcollection | feedback_id | 365 days |
| feedback/{uid}/exercise_outcomes | Subcollection | outcome_id | 365 days |
| exercise_sessions/{uid}/sessions | Subcollection | session_id | 365 days |
| exercise_sessions/{uid}/tasks | Subcollection | task_id | 365 days |
| soundscape_sessions/{uid}/sessions | Subcollection | session_id | 180 days |
| safety_audit/{uid}/events | Subcollection | event_id | 730 days |
| health_metrics/{uid}/daily | Subcollection | date | 365 days |
| bandit_state | Single doc | user_id | Until deletion |
| interaction_metrics | Single doc | user_id | 365 days |
| consent_records | Single doc | user_id | Until deletion |
| training_examples | Flat, query | doc_id + user_id | 365 days |
| model_performance | Flat, query | doc_id + user_id | 365 days |
| finetuning_jobs | Flat | job_id | Until deletion |
| escalation_queue | Flat, query | escalation_id + user_id | Until deletion |
| annual_reviews/{uid}/reviews | Subcollection | review_id | Until deletion |
| audit_log | Flat | entry_id | 2555 days (7yr) |

---

## 7. Prompt Engine & Personality System

### How the System Prompt is Built

The `PromptEngine` (`prompt_engine.py`) builds the system prompt in layers:

```
Final System Prompt = [
    User Memories (personal context)
    + User Profile Summary (demographics, preferences)
    + Conversation Summary (previous context)
    + Base Persona Prompt (communication style)
    + RAG Context (retrieved knowledge)
    + Active Exercise Context
    + Due Practice Tasks
    + Effectiveness Insights
    + Soundscape Suggestion
    + Cultural Context
    + Health Context (wearable data)
    + Dependency Override (if triggered)
    + Safety Override (if triggered - HIGHEST PRIORITY)
    + Intent Modifier (crisis, venting, reflecting, etc.)
    + Emotion Tone Hint (sad, anxious, etc.)
]
```

### Communication Styles

| Style | Personality | When Used |
|-------|------------|-----------|
| empathetic | Warm, compassionate, validating | Default |
| direct | Straightforward, action-oriented | User preference |
| analytical | Evidence-based, reasoning-focused | User preference |
| casual | Friendly, relatable, informal | User preference |

### Intent Modifiers

The prompt adjusts based on detected intent:
- **crisis** - Safety-first, provide resources, no minimizing
- **venting** - Active listening, validation, no advice unless asked
- **seeking_advice** - Practical, actionable suggestions
- **reflecting** - Open-ended questions, deeper exploration
- **doing_exercise** - Guide through exercise steps

---

## 8. Safety & Crisis Pipeline

### Three-Layer Safety System

**Layer 1: Input Screening** (pre-LLM, no API call)
- Keyword matching for crisis indicators (suicide, self-harm, etc.)
- Jailbreak detection (prompt injection attempts)
- Risk level assignment: LOW, MEDIUM, HIGH, CRITICAL
- CRITICAL triggers immediate crisis response (bypasses LLM entirely)

**Layer 2: Output Validation** (post-LLM)
- Screens generated response for harmful content
- Modifies response if unsafe patterns detected
- Adds safety disclaimers when appropriate

**Layer 3: Auto-Escalation** (Phase 20)
- CRITICAL safety event -> URGENT escalation ticket
- HIGH dependency score (>= 61) -> HIGH escalation ticket
- 3+ HIGH safety events in 7 days -> HIGH escalation ticket (repeated pattern)
- Dedup: skips if user has active PENDING/ACKNOWLEDGED/IN_PROGRESS ticket

### Crisis Response Flow

```
User message contains crisis keywords
    |
    v
SafetyService.check_input() -> CRITICAL
    |
    v
SafetyService.get_crisis_intercept_response()
    -> Returns pre-built helpline message (NO LLM call)
    |
    v
Log safety event to safety_audit/{user_id}/events
    |
    v
EscalationService.check_and_create_escalation()
    -> Creates URGENT ticket in escalation_queue
    |
    v
Return crisis response to user immediately
```

---

## 9. Reinforcement Learning System

### Thompson Sampling for Exercise Recommendations

The RL service (`rl_service.py`) uses Thompson Sampling to learn which therapy exercises work best for each user's emotional state.

**Key Concepts:**
- **Arm**: An `emotion|technique` pair (e.g., `sad|gratitude_journal`)
- **Bandit State**: Per-user document in `bandit_state/{user_id}` with successes/failures per arm
- **Success**: Exercise outcome with helpfulness >= 4 (out of 5)
- **Exploration Bonus**: Configurable (default 0.5) added to Thompson Sampling score

**Flow:**
1. User starts exercise with detected emotion
2. RL service samples from Beta distribution for each arm matching that emotion
3. Arm with highest sample score is recommended
4. After exercise completion, user rates helpfulness
5. Outcome updates the arm's success/failure counts

**Minimum Data Requirement:** RL only activates after 3+ outcomes per arm. Before that, falls back to rule-based effectiveness scoring.

---

## 10. Fine-Tuning Pipeline

### Training Data Extraction

The `FineTuningService` (`finetuning_service.py`) extracts high-quality training data from user conversations:

1. **Consent Check**: Only uses data from users who consented to `ML_TRAINING`
2. **Quality Filter**: Only includes conversations with feedback rating >= "helpful" and helpfulness score >= 4
3. **JSONL Format**: Converts conversations to OpenAI fine-tuning format
4. **Storage**: Saves to `training_examples/{doc_id}` in Firestore

### A/B Testing

When a fine-tuned model is active (`FT_MODEL_ID` set):
- User routing: MD5 hash of user_id determines base vs fine-tuned (configurable split %)
- Performance logged to `model_performance/{doc_id}`
- Stats endpoint compares latency, token usage, feedback scores between models

### Job Management

- `POST /finetuning/submit-job` submits to OpenAI fine-tuning API
- `GET /finetuning/jobs` lists all jobs with status
- `GET /finetuning/ab-stats` compares base vs fine-tuned model performance

---

## 11. GDPR Compliance Layer

### Data Export (Article 20)

`compliance_service.py` -> `export_user_data(user_id)`:
- Queries all 22 collections for user data
- Returns a `DataExportResponse` with all user data serialized
- Logs export to audit trail

### Cascade Deletion (Article 17)

`compliance_service.py` -> `delete_all_user_data(user_id)`:
- Deletes data from ALL collections in order:
  1. Subcollections (memories, feedback, exercises, soundscapes, safety, health, annual_reviews)
  2. Parent documents for subcollection containers
  3. Flat collections via query (chat_sessions, training_examples, model_performance, escalation_queue)
  4. Single documents (user_profiles, bandit_state, interaction_metrics, consent)
  5. Clears in-memory caches
- Returns a `DeletionReceipt` with counts per collection
- Logs deletion to audit trail
- Does NOT delete audit_log entries (legal requirement)

### When Adding New Collections

You MUST update both methods in `compliance_service.py`:
1. `export_user_data()` - Add data reading logic
2. `delete_all_user_data()` - Add deletion logic
3. `DataExportResponse` in `models.py` - Add field for the new data

---

## 12. Middleware Stack

Three middleware layers applied to every request (defined in `middleware.py`):

```
Request -> RateLimitMiddleware -> MetricsMiddleware -> PrivacyMiddleware -> Route Handler
```

### RateLimitMiddleware

- Token bucket algorithm (thread-safe)
- Separate limits for `/chat*` endpoints vs global
- Configurable: `RATE_LIMIT_CHAT` (10/min), `RATE_LIMIT_GLOBAL` (100/min)
- Returns 429 Too Many Requests when exceeded

### MetricsMiddleware

- Tracks per-endpoint latency (p50, p95, p99)
- Request counts and error rates
- UUID normalization in paths for clean grouping
- Exposes data via `GET /metrics` and `GET /admin/dashboard`

### PrivacyMiddleware

- Strips sensitive data from logs
- Ensures PII is not leaked in error responses

---

## 13. Caching Strategy

### TTLCache (cache.py)

In-memory cache with time-to-live eviction. No Redis dependency.

```python
cache = get_cache()
cache.set("key", value, ttl=60)     # Cache for 60 seconds
result = cache.get("key")            # Returns None if expired
cache.invalidate("key")              # Manual invalidation
cache.invalidate_prefix("user:123")  # Invalidate all keys with prefix
```

### Cache TTLs by Data Type

| Data | TTL | Config Key |
|------|-----|-----------|
| User profiles | 60s | CACHE_TTL_USER_PROFILE |
| Effectiveness scores | 120s | CACHE_TTL_EFFECTIVENESS |
| Emotion detection | 300s | CACHE_TTL_EMOTION |
| RL bandit state | 300s | RL_CACHE_TTL |
| Dashboard metrics | 60s | DASHBOARD_CACHE_TTL |
| Max cache entries | 500 | CACHE_MAX_SIZE |

### Cache Invalidation

Caches are invalidated when:
- User profile is updated (`cache.invalidate(user_profile_key(user_id))`)
- User data is deleted (GDPR cascade deletion)
- Effectiveness is recomputed after new feedback

---

## 14. File-by-File Reference

### Core Application

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | ~4,200 | FastAPI app, 67+ endpoints, /chat pipeline, startup |
| `models.py` | ~1,250 | 70+ Pydantic models, 12 enums |
| `config.py` | ~240 | 68 config fields, env var loading, singleton |

### Service Files

| File | Lines | Key Methods |
|------|-------|-------------|
| `firebase_service.py` | ~200 | `db` property, `store_chat_session()`, `get_chat_session()` |
| `user_service.py` | ~300 | `get_or_create_profile()`, `update_profile()`, `append_mood_entry()` |
| `emotion_service.py` | ~200 | `detect_emotion()` -> `EmotionDetectionResult` |
| `memory_service.py` | ~350 | `store_memory()`, `search_memories()`, `consolidate_memories()` |
| `chat_service.py` | ~150 | Core chat logic (mostly moved to chat_agent_service) |
| `chat_agent_service.py` | ~800 | LangChain agent with therapy/memory/soundscape tools |
| `prompt_engine.py` | ~400 | `build_system_prompt()` with 5-layer architecture |
| `therapy_service.py` | ~700 | `get_exercises()`, `start_session()`, `advance_session()` |
| `progress_service.py` | ~200 | `get_progress_summary()`, task management |
| `feedback_service.py` | ~400 | `submit_feedback()`, `compute_effectiveness()` |
| `soundscape_service.py` | ~400 | `get_soundscapes()`, `start_session()`, `get_recommendation()` |
| `safety_service.py` | ~400 | `check_input()`, `validate_output()`, `get_crisis_intercept_response()` |
| `dependency_service.py` | ~400 | `assess_dependency()` (7 signals), `record_interaction()` |
| `cultural_service.py` | ~250 | `extract_context()`, `check_bias()` |
| `compliance_service.py` | ~650 | `export_user_data()`, `delete_all_user_data()`, retention enforcement |
| `audit_service.py` | ~150 | `log()` -> writes to `audit_log` collection |
| `storage_service.py` | ~100 | `generate_signed_url()` for GCS audio files |
| `voice_service.py` | ~250 | `text_to_speech()`, `speech_to_text()` |
| `wearable_service.py` | ~400 | `sync_health_data()`, `get_health_summary()` |
| `rl_service.py` | ~300 | `get_recommendation()`, `update_outcome()` (Thompson Sampling) |
| `finetuning_service.py` | ~680 | `extract_training_data()`, `submit_job()`, `route_model()` |
| `monitoring_service.py` | ~400 | `get_system_metrics()`, `get_all_metrics()` |
| `escalation_service.py` | ~450 | `check_and_create_escalation()`, `generate_annual_review()` |

### Infrastructure

| File | Lines | Purpose |
|------|-------|---------|
| `cache.py` | ~100 | TTLCache with thread-safe operations |
| `middleware.py` | ~250 | Rate limiting, metrics collection, privacy |
| `start_local.py` | ~20 | Local dev server launcher (uvicorn) |

---

## 15. How To: Add a New Feature (Phase)

Follow these steps for any new feature addition:

### Step 1: Configuration (`config.py`)

Add fields to `AppConfig` class (after the last section):
```python
# My Feature (Phase N)
MY_FEATURE_ENABLED: bool = True
MY_FEATURE_THRESHOLD: int = 10
```

Add to `_load_config()` return statement:
```python
MY_FEATURE_ENABLED=_env_bool("MY_FEATURE_ENABLED", True),
MY_FEATURE_THRESHOLD=_env_int("MY_FEATURE_THRESHOLD", 10),
```

### Step 2: Models (`models.py`)

Add new Pydantic models at the end of the file:
```python
# ── Phase N: My Feature ───────

class MyRequest(BaseModel):
    field: str = ""

class MyResponse(BaseModel):
    result: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
```

### Step 3: Service (`my_service.py`)

Create a new service file following the singleton pattern (see Section 2).

### Step 4: Wire into main.py

**4a. Import** (at top of main.py):
```python
from my_service import get_my_service
```

**4b. Model imports** (in the `from models import (...)` block):
```python
MyRequest, MyResponse,
```

**4c. Startup** (in the `services = {...}` dict):
```python
"my_feature": get_my_service,
```

**4d. OpenAI injection** (if needed, after finetuning/escalation injection):
```python
try:
    _my_svc = get_my_service()
    _my_svc.set_openai_client(openai_client)
except Exception as _err:
    print(f"Warning: Could not inject OpenAI client: {_err}")
```

**4e. Endpoints** (before exception handlers):
```python
@app.get("/my-feature/{user_id}")
async def get_my_feature(user_id: str):
    try:
        svc = get_my_service()
        result = svc.get_data(user_id)
        return JSONResponse(content={"status": "success", "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 5: GDPR Compliance (`compliance_service.py`)

If your feature stores user data:
- Add export logic in `export_user_data()`
- Add deletion logic in `delete_all_user_data()`
- Add field to `DataExportResponse` in `models.py`

### Step 6: Environment Template (`.env.example`)

Add your new config vars with a section header:
```env
# My Feature (Phase N)
MY_FEATURE_ENABLED=true
MY_FEATURE_THRESHOLD=10
```

### Step 7: Verification Tests

Write tests covering:
- Config defaults and env overrides
- Model construction and serialization
- Service graceful degradation (no DB/no OpenAI)
- Endpoint presence in main.py source
- GDPR export/deletion coverage

---

## 16. How To: Add a New API Endpoint

1. Choose the HTTP method and path
2. Define request/response models in `models.py`
3. Add the endpoint in `main.py` **before the exception handlers**
4. Follow this pattern:

```python
@app.post("/my-endpoint/{user_id}")
async def my_endpoint(user_id: str, request: MyRequest):
    """Endpoint docstring (shows in /docs)."""
    try:
        svc = get_my_service()
        result = svc.do_something(user_id, request.field)
        if result is None:
            raise HTTPException(status_code=404, detail="Not found")
        return JSONResponse(content={"status": "success", "data": result})
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"My endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Important:** If you have both a fixed path and a dynamic path with the same prefix, the fixed path MUST come first:
```python
@app.get("/items/stats")          # FIRST (fixed)
@app.get("/items/{item_id}")      # SECOND (dynamic)
```

---

## 17. How To: Add a New Firestore Collection

1. **Define the collection constant** in your service class:
   ```python
   MY_COLLECTION = "my_collection"
   ```

2. **Choose the pattern:**
   - Single doc per user: `my_collection/{user_id}`
   - Subcollection: `my_collection/{user_id}/items/{item_id}`
   - Flat with query: `my_collection/{doc_id}` with `user_id` field

3. **GDPR compliance** (required for any collection storing user data):
   - Add to `export_user_data()` in `compliance_service.py`
   - Add to `delete_all_user_data()` in `compliance_service.py`
   - Add field to `DataExportResponse` in `models.py`

4. **For subcollections**, use the existing helper methods:
   ```python
   # Export
   data = self._read_subcollection(f"my_collection/{user_id}", "items")

   # Deletion
   counts["my_items"] = self._delete_subcollection(f"my_collection/{user_id}", "items")
   self._delete_parent_doc("my_collection", user_id)
   ```

5. **For flat collections**, use query-based patterns:
   ```python
   # Export
   docs = self.db.collection("my_collection").where("user_id", "==", user_id).stream()

   # Deletion
   docs = self.db.collection("my_collection").where("user_id", "==", user_id).stream()
   for doc in docs:
       doc.reference.delete()
       count += 1
   ```

---

## 18. How To: Add a New Config Field

1. Add to `AppConfig` class in `config.py`:
   ```python
   MY_FIELD: int = 42  # descriptive comment
   ```

2. Add to `_load_config()` in `config.py`:
   ```python
   MY_FIELD=_env_int("MY_FIELD", 42),
   ```

3. Add to `.env.example`:
   ```env
   MY_FIELD=42
   ```

4. Use it in services:
   ```python
   config = get_config()
   value = config.MY_FIELD
   ```

**Note:** The `AppConfig` dataclass is frozen (immutable). You cannot modify config values after initialization. To test with different values, create a new `AppConfig` instance.

---

## 19. How To: Integrate into the /chat Pipeline

To add processing to the main chat flow:

### Adding Pre-LLM Processing

Insert your code after the dependency detection block in both `/chat` and `/chat/stream`:

```python
# In /chat endpoint, after dependency detection:
# Phase N: My Feature
my_context_text = ""
if request.user_id:
    try:
        my_svc = get_my_service()
        my_result = my_svc.process(request.user_id, prompt)
        if my_result:
            my_context_text = my_svc.format_for_prompt(my_result)
    except Exception as e:
        logger.warning(f"My feature failed, continuing: {e}")
```

Then pass `my_context_text` to the prompt engine.

### Adding Post-LLM Processing

Insert after the safety event log block:

```python
# Phase N: My Feature post-processing
if request.user_id:
    try:
        my_svc = get_my_service()
        my_svc.process_response(request.user_id, bot_response)
    except Exception:
        pass  # Never break the chat pipeline
```

### Critical Rules

1. **Always wrap in try/except** -- your feature must NEVER crash the /chat endpoint
2. **Add to BOTH endpoints** -- `/chat` and `/chat/stream` have parallel pipelines
3. **Use `pass` in except blocks** -- log errors but don't propagate
4. **Test with `'variable_name' in dir()`** -- variables from try blocks may not exist

---

## 20. Common Pitfalls

### 1. Circular Imports

**Problem:** Service A imports Service B which imports Service A.
**Solution:** Use lazy imports inside methods:
```python
def my_method(self):
    from other_service import get_other_service  # Lazy import
    svc = get_other_service()
```

### 2. Variable Scoping in /chat Pipeline

**Problem:** Variables defined in try blocks may not exist when referenced later.
**Solution:** Initialize before the try block or check with `'var' in dir()`:
```python
my_result = None
try:
    my_result = my_service.compute()
except Exception:
    pass

# Later...
if my_result is not None:
    do_something(my_result)
```

### 3. Frozen Config Modification

**Problem:** Trying to modify `AppConfig` fields at runtime.
**Solution:** `AppConfig` is a frozen dataclass. Create a new instance for testing:
```python
test_config = AppConfig(MY_FIELD=99)  # New instance with overrides
```

### 4. Forgetting GDPR Compliance

**Problem:** New collection stores user data but export/deletion don't cover it.
**Solution:** Always update `compliance_service.py` AND `DataExportResponse` when adding user data.

### 5. Endpoint Path Ordering

**Problem:** Dynamic path `/{id}` catches static paths like `/stats`.
**Solution:** Always define static paths BEFORE dynamic paths:
```python
@app.get("/items/stats")       # FIRST
@app.get("/items/{item_id}")   # SECOND
```

### 6. Windows File Encoding

**Problem:** Unicode characters in test output fail on Windows.
**Solution:** Add at the top of test files:
```python
sys.stdout.reconfigure(encoding='utf-8')
```

### 7. Missing OpenAI Client

**Problem:** Service methods fail because OpenAI client wasn't injected.
**Solution:** Always check `self._openai_client` before using it:
```python
if not self._openai_client:
    logger.warning("OpenAI client not available")
    return None
```

---

## 21. Testing Conventions

### Test File Structure

Each phase has its own test file: `test_phaseNN.py`

```python
import sys
import os
import unittest

sys.stdout.reconfigure(encoding='utf-8')

# Prevent real API calls
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

class TestPhaseNNConfig(unittest.TestCase):
    """Config field defaults and env overrides."""
    ...

class TestPhaseNNModels(unittest.TestCase):
    """Model construction and serialization."""
    ...

class TestPhaseNNService(unittest.TestCase):
    """Service graceful degradation without DB."""
    ...

class TestPhaseNNMainIntegration(unittest.TestCase):
    """Verify main.py has correct imports, endpoints, wiring."""
    # Reads main.py source and checks for string presence
    ...

class TestPhaseNNCompliance(unittest.TestCase):
    """GDPR export/deletion coverage."""
    ...
```

### What to Test

| Category | What | How |
|----------|------|-----|
| Config | Defaults correct | `self.assertEqual(AppConfig().FIELD, expected)` |
| Config | Env override works | Set env var, call `_load_config()`, verify |
| Models | Construction with defaults | `Model()` doesn't crash |
| Models | All fields present | `set(model.model_dump().keys()) == expected` |
| Models | Serialization | `model.model_dump()` returns valid dict |
| Service | Init without DB | Service constructs with `_db = None` |
| Service | Methods return defaults | `svc.list_items()` returns `[]` |
| Service | Singleton works | `get_svc() is get_svc()` |
| Main.py | Import present | `assertIn("from xxx import", source)` |
| Main.py | Startup registered | `assertIn('"name": get_service', source)` |
| Main.py | Endpoints exist | `assertIn('"/path"', source)` |
| GDPR | Export covered | `assertIn("collection_name", compliance_source)` |
| GDPR | Deletion covered | `assertIn("collection_name", compliance_source)` |

### Running Tests

```bash
# All tests
python -m pytest -v

# Specific phase
python -m pytest test_phase20.py -v

# Syntax check
python -c "import py_compile; py_compile.compile('main.py', doraise=True)"
```

---

## 22. Deployment Notes

### Google Cloud Run

- **Image**: Python 3.12-slim Docker container
- **Port**: 8080 (hardcoded in Dockerfile and app config)
- **Scaling**: 0-10 instances (configurable)
- **Startup**: All services initialize during FastAPI startup event
- **Health check**: `GET /health` returns 200 when ready

### Environment Secrets

- `OPENAI_API_KEY` should be stored in Google Secret Manager
- Firebase service account key should be mounted as a file
- All other config vars can be set as Cloud Run env vars

### Build & Deploy

```bash
# CI/CD via Cloud Build
gcloud builds submit --config cloudbuild.yaml

# Manual deploy
gcloud run deploy lucille-llm \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

### Monitoring

- **Logs**: Google Cloud Logging (structured JSON format)
- **Dashboard**: `GET /admin/dashboard` (HTML with Chart.js)
- **Metrics**: `GET /metrics` and `GET /admin/dashboard/all`
- **Alerts**: Set up based on escalation queue depth and error rates

### Performance Notes

- The first request after cold start takes ~5-10 seconds (service initialization)
- FAISS index loading is the slowest startup operation (~2-3 seconds)
- Subsequent requests: typically 1-3 seconds (depends on OpenAI latency)
- Dashboard metrics are cached (60s TTL) to avoid expensive aggregation queries
- All Firestore queries use `.limit()` to cap costs

---

## Quick Reference Card

### Adding a Complete Feature

```
1. config.py      -> Add fields to AppConfig + _load_config()
2. models.py      -> Add request/response models
3. *_service.py   -> Create service (singleton pattern)
4. main.py        -> Import, startup, inject OpenAI, add endpoints
5. compliance.py  -> GDPR export + deletion (if user data)
6. .env.example   -> Add config vars
7. test_*.py      -> Write verification tests
```

### File Locations for Common Changes

| Change | File | Location |
|--------|------|----------|
| New config field | config.py | AppConfig class + _load_config() |
| New model | models.py | End of file |
| New service | new_service.py | Create new file |
| New endpoint | main.py | Before exception handlers |
| GDPR for new data | compliance_service.py | export + delete methods |
| New env var | .env.example | End of file |
| Chat pipeline change | main.py | Search for "Phase" comments |
| Dashboard metric | monitoring_service.py | Appropriate get_*_metrics() |
| Safety rule | safety_service.py | check_input() or validate_output() |
| Prompt change | prompt_engine.py | build_system_prompt() |
