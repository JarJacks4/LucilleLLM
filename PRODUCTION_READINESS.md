# LucilleLLM Production Readiness Plan

**Rating: 7.5/10** | **Target: 9/10** | **Created: 2026-03-29**

---

## Critical Priority

### 1. Add Authentication & Authorization

**Problem:** All 67+ endpoints are publicly accessible. Anyone can delete user data, access admin dashboard, submit fine-tuning jobs, or read other users' sessions.

**Files to change:** `main.py`, new `auth_middleware.py`

**Steps:**
- Add Firebase Auth verification middleware
- Create a `get_current_user()` dependency that validates JWT tokens
- Protect admin endpoints (`/admin/*`, `/finetuning/*`) with role-based access (admin role)
- Ensure users can only access their own data — validate `user_id` in path matches authenticated user
- Add API key auth option for service-to-service calls

**Example:**
```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin.auth as firebase_auth

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        decoded = firebase_auth.verify_id_token(credentials.credentials)
        return decoded
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

async def require_admin(user=Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# Usage:
@app.get("/admin/dashboard", dependencies=[Depends(require_admin)])
async def dashboard(): ...

@app.get("/users/{user_id}/memories")
async def get_memories(user_id: str, user=Depends(get_current_user)):
    if user["uid"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    ...
```

---

### 2. Add Test Suite

**Problem:** No unit or integration tests. For a mental health app handling crisis detection and data deletion, untested code is a liability.

**Files to create:** `tests/` directory

**Steps:**
- Install pytest: add `pytest>=7.0`, `pytest-asyncio>=0.21`, `httpx>=0.27` to requirements.txt
- Create test structure:

```
tests/
  conftest.py              # Shared fixtures (mock Firebase, mock OpenAI)
  test_safety_service.py   # Crisis keyword detection, jailbreak detection, output validation
  test_dependency_service.py  # 7 dependency signals, risk scoring
  test_emotion_service.py  # Emotion/intent parsing
  test_compliance_service.py  # GDPR cascade deletion, data export
  test_therapy_service.py  # Exercise templates, session flow
  test_middleware.py       # Rate limiting logic
  test_prompt_engine.py    # 5-layer prompt assembly
  test_rl_service.py       # Thompson Sampling arm selection
  test_memory_service.py   # Memory CRUD, semantic search
  test_endpoints.py        # FastAPI TestClient integration tests
```

**Priority test cases:**
```python
# test_safety_service.py
def test_critical_crisis_detected():
    """'I want to kill myself' must trigger CRITICAL level"""

def test_moderate_crisis_detected():
    """'I feel hopeless' must trigger MODERATE level"""

def test_jailbreak_blocked():
    """'Ignore all instructions' must be flagged"""

def test_clean_input_passes():
    """Normal therapeutic text must not trigger false positives"""

# test_compliance_service.py
def test_gdpr_cascade_deletion():
    """Deletion must hit all 22 Firestore collections"""

def test_data_export_completeness():
    """Export must include all user data categories"""

# test_dependency_service.py
def test_high_frequency_detection():
    """>20 messages/hour must raise dependency signal"""

def test_normal_usage_clean():
    """5 messages/hour must return NONE risk"""
```

**Target:** 70%+ code coverage on all service files. Add `pytest` to CI in `cloudbuild.yaml`.

---

## High Priority

### 3. Split `main.py` into APIRouters

**Problem:** `main.py` is 4,207 lines — a god file containing all 67+ endpoints, middleware setup, and orchestration logic.

**Steps:**
- Create `routes/` directory with separate router files
- Move endpoints into their respective routers
- Keep `main.py` as the app factory (startup, middleware, router registration)

**Target structure:**
```
routes/
  __init__.py
  chat.py          # /chat, /chat/stream, /chat/voice, /chat/{session_id}
  users.py         # /users/*, /users/{user_id}/mood, /users/{user_id}/sessions
  therapy.py       # /therapy/*, exercises, sessions, progress
  feedback.py      # /feedback/*
  soundscapes.py   # /soundscapes/*
  voice.py         # /tts, /stt
  wearables.py     # /wearables/*
  safety.py        # /safety/*
  compliance.py    # /users/{user_id}/consent, /users/{user_id}/export, /users/{user_id}/data
  finetuning.py    # /finetuning/*
  rl.py            # /rl/*
  admin.py         # /admin/*
  reviews.py       # /reviews/*
  health.py        # /, /health, /health/detailed, /metrics
```

**Example:**
```python
# routes/chat.py
from fastapi import APIRouter, Depends
router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("")
async def chat(request: ChatRequest): ...

@router.post("/stream")
async def chat_stream(request: ChatRequest): ...

# main.py (after refactor)
from routes import chat, users, therapy, admin, ...

app = FastAPI(title="LucilleLLM")
app.include_router(chat.router)
app.include_router(users.router)
app.include_router(therapy.router)
app.include_router(admin.router)
```

**main.py should shrink to ~200 lines** (app factory + middleware + startup).

---

### 4. Fix Distributed Rate Limiting

**Problem:** Rate limiting in `middleware.py` uses in-memory dicts. With Cloud Run scaling to 10 instances, each instance has its own counter. A user can send 10x the limit by hitting different instances.

**Options (pick one):**

**Option A: Redis (Memorystore) — recommended**
```python
import redis.asyncio as redis

r = redis.from_url(os.getenv("REDIS_URL"))

async def check_rate_limit(user_id: str, limit: int, window: int) -> bool:
    key = f"rate:{user_id}:{int(time.time()) // window}"
    count = await r.incr(key)
    if count == 1:
        await r.expire(key, window)
    return count <= limit
```

**Option B: Cloud Armor rate policies (no code change)**
- Configure rate limiting at the load balancer level in GCP Console
- Simpler but less granular (IP-based, not user-based)

**Option C: Accept per-instance limits (if acceptable)**
- Document that rate limits are approximate
- Only viable if abuse prevention isn't critical

---

### 5. Fix In-Memory Cache for Multi-Instance

**Problem:** `cache.py` is a local dict. Each Cloud Run instance has its own cache — no sharing, no invalidation.

**Steps:**
- For short TTLs (60s user profile, 120s effectiveness): keep as-is, inconsistency is acceptable
- For rate limiting and session state: move to Redis (see #4 above)
- Add cache-aside pattern with Redis for longer-lived data:

```python
# cache.py enhancement
class DistributedCache:
    def __init__(self, redis_url: str = None):
        self._local = TTLCache()  # existing in-memory
        self._redis = redis.from_url(redis_url) if redis_url else None

    async def get(self, key: str):
        # Check local first
        val = self._local.get(key)
        if val: return val
        # Fall back to Redis
        if self._redis:
            val = await self._redis.get(key)
            if val:
                self._local.set(key, val)
                return val
        return None
```

---

## Medium Priority

### 6. Add Retry Logic for External Services

**Problem:** OpenAI, Firebase, GCS calls can fail transiently (timeouts, 429s, 503s). No retry logic exists.

**Steps:**
- Add `tenacity` to requirements.txt
- Wrap OpenAI calls with retries:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError, APIConnectionError

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError))
)
async def call_openai(client, **kwargs):
    return await client.chat.completions.create(**kwargs)
```

- Wrap Firebase calls similarly for `google.api_core.exceptions.ServiceUnavailable`
- Add circuit breaker pattern for sustained failures (optional, use `pybreaker`)

---

### 7. Add Input Sanitization

**Problem:** Some endpoints pass user strings directly into prompts without validation beyond Pydantic types.

**Steps:**
- Add max length validation to all text input fields in `models.py`:
```python
class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)
    session_id: Optional[str] = Field(None, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
```

- Add a sanitization utility:
```python
# utils/sanitize.py
import re

def sanitize_input(text: str, max_length: int = 4000) -> str:
    text = text[:max_length]
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # strip control chars
    return text.strip()
```

- Apply to all user-facing text inputs before processing

---

### 8. Move FAISS Index to External Storage

**Problem:** `embeddings_data.pkl` (53MB) and `texts.pkl` (3.3MB) are baked into the Docker image. Updating knowledge requires a full redeploy.

**Options:**

**Option A: Load from GCS at startup**
```python
# chat_service.py
from google.cloud import storage

def load_faiss_index():
    if os.getenv("FAISS_GCS_BUCKET"):
        client = storage.Client()
        bucket = client.bucket(os.getenv("FAISS_GCS_BUCKET"))
        bucket.blob("faiss/embeddings_data.pkl").download_to_filename("/tmp/embeddings_data.pkl")
        bucket.blob("faiss/texts.pkl").download_to_filename("/tmp/texts.pkl")
        return load_from("/tmp/")
    return load_from("faiss_vecdb/")  # fallback to local
```

**Option B: Managed vector DB (longer term)**
- Migrate to Pinecone, Weaviate, or Vertex AI Vector Search
- Enables dynamic document CRUD without redeployment
- Better for scaling beyond current knowledge base size

---

### 9. Add Global Error Handling

**Problem:** Errors return inconsistent formats — some raw strings, some dicts, some default FastAPI 422s.

**Steps:**
```python
# error_handlers.py
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "code": "VALIDATION_ERROR",
            "details": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": "INTERNAL_ERROR"
        }
    )

class AppError(Exception):
    def __init__(self, message: str, code: str, status: int = 400):
        self.message = message
        self.code = code
        self.status = status

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status,
        content={"error": exc.message, "code": exc.code}
    )
```

---

## Low Priority

### 10. Add CORS Middleware

**Problem:** No CORS configuration. Browser-based frontends will fail with cross-origin errors.

**Fix:**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "").split(","),  # e.g. "https://app.lucille.ai"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Add to `config.py`:
```python
CORS_ORIGINS: str = ""  # comma-separated allowed origins
```

---

### 11. Enhance Health Checks

**Problem:** `/health` returns basic status but doesn't verify dependency connectivity.

**Fix:**
```python
@app.get("/health/detailed")
async def detailed_health():
    checks = {}

    # Firebase
    try:
        db = firestore.client()
        db.collection("health_check").limit(1).get()
        checks["firebase"] = "healthy"
    except Exception as e:
        checks["firebase"] = f"unhealthy: {e}"

    # OpenAI
    try:
        client = openai.AsyncOpenAI()
        await client.models.list()
        checks["openai"] = "healthy"
    except Exception as e:
        checks["openai"] = f"unhealthy: {e}"

    # FAISS
    try:
        checks["faiss"] = "healthy" if faiss_index is not None else "not loaded"
    except Exception as e:
        checks["faiss"] = f"unhealthy: {e}"

    overall = "healthy" if all(v == "healthy" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks, "uptime": get_uptime()}
```

---

### 12. Add Structured Logging

**Problem:** Mixed `print()` and `logging` calls. Cloud Run expects JSON logs for proper log level filtering.

**Fix:**
```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        })

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)
```

Replace all `print()` statements with `logger.info()` / `logger.error()`.

---

### 13. Add `.env` Safety Check

**Problem:** Local development uses `.env` files which could leak API keys if committed.

**Steps:**
- Verify `.env` is in `.gitignore`
- Add `.env.example` with placeholder values:
```
OPENAI_API_KEY=sk-your-key-here
GOOGLE_CLOUD_PROJECT=your-project-id
FIREBASE_CREDENTIALS_PATH=path/to/serviceAccount.json
REDIS_URL=redis://localhost:6379
CORS_ORIGINS=http://localhost:3000
```
- Add a pre-commit hook to block `.env` files from being committed

---

### 14. Add CI Testing to Cloud Build

**Problem:** `cloudbuild.yaml` only builds and deploys — no test step.

**Fix:** Add test step before deployment:
```yaml
steps:
  # Run tests
  - name: 'python:3.12-slim'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio httpx
        python -m pytest tests/ -v --tb=short
    id: 'test'

  # Build (existing step)
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '...']
    waitFor: ['test']

  # Deploy (existing step)
  ...
```

---

## Implementation Order

| Order | Task | Effort | Impact |
|-------|------|--------|--------|
| 1 | Add authentication (#1) | 1-2 days | Critical security fix |
| 2 | Split main.py into routers (#3) | 1 day | Unblocks all other work |
| 3 | Add test suite (#2) | 2-3 days | Safety net for everything else |
| 4 | Global error handling (#9) | 2 hours | Quick win, consistent API |
| 5 | CORS middleware (#10) | 15 mins | Quick win for frontend |
| 6 | Input sanitization (#7) | 2 hours | Security hardening |
| 7 | Retry logic (#6) | 3 hours | Reliability |
| 8 | Structured logging (#12) | 2 hours | Observability |
| 9 | Health check enhancement (#11) | 1 hour | Operations |
| 10 | CI testing (#14) | 1 hour | Quality gate |
| 11 | Distributed rate limiting (#4) | 1 day | Scaling fix |
| 12 | Distributed cache (#5) | 1 day | Scaling fix |
| 13 | External FAISS index (#8) | Half day | Operational flexibility |
| 14 | .env safety (#13) | 30 mins | Security hygiene |

---

## Summary

**Current state:** Strong domain logic, weak infrastructure. The therapy engine, safety system, RL recommendations, compliance, and 5-layer personality are well designed. The gaps are all in operational concerns — auth, testing, error handling, and distributed state.

**After all fixes:** Production-grade mental health AI with proper security, testing, observability, and scalability. Rating target: **9/10**.
