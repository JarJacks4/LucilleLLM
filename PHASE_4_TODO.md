# Phase 4 — Future improvements (not yet started)

These are the items deferred from the Phase 1–3 hardening sweep. None are
launch-blockers; they're code-quality and scaling improvements to take on
once Phases 1–3 are deployed and validated in production.

---

## 1. Refactor `main.py` into route modules ⚠️ Largest item

**Current state:** `main.py` is **~4,550 lines** containing 77 endpoints,
the OpenAI client init, the structured logger, the FAISS index loader, and
all the request lifecycle glue. This is a maintainability red flag — MAANG
standard is 200–500 lines per file.

**Why deferred from Phase 3:** This is a 1–2 day refactor on its own. Every
endpoint needs to move, imports need to be redistributed, and the change
touches every test that imports `main`. Doing it in the same session as the
auth/async/RBAC work would have made the diff impossible to review.

**Target structure:**

```
main.py                  ~150 lines: app init, middleware, lifespan, exception handlers
routes/
├── __init__.py
├── chat.py              POST /chat, /chat/stream, /chat/voice
├── sessions.py          GET/DELETE /chat/{session_id}, /sessions
├── users.py             /users/* (onboard, profile, mood, sessions)
├── memories.py          /users/{uid}/memories/*
├── therapy.py           /therapy/* (exercises, tasks, progress)
├── feedback.py          /feedback/* (response, outcomes, effectiveness)
├── soundscapes.py       /soundscapes/*
├── safety.py            /safety/*
├── compliance.py        /users/{uid}/export, /consent, GDPR
├── voice.py             /tts, /stt
├── wearables.py         /wearables/*
├── assessments.py       /assessments/*
├── reviews.py           /reviews/*
├── admin.py             /admin/* (retention, audit, dashboard, escalations)
└── finetuning.py        /finetuning/*
dependencies.py          shared FastAPI dependencies (services, clients)
```

**Migration approach (do one route module at a time, deploy in between):**

1. Create `routes/chat.py`
2. Move the `/chat`, `/chat/stream`, `/chat/voice` handlers there
3. Use `APIRouter()` and `app.include_router(chat.router)` in `main.py`
4. Run tests, deploy, verify
5. Repeat for the next module

**Critical files that should move with the refactor:**
- `main.py:751-988` → `routes/chat.py` (POST /chat handler)
- `main.py:1230-1500` → `routes/chat.py` (POST /chat/stream handler)
- `main.py:3193+` → `routes/chat.py` (POST /chat/voice handler)
- `main.py:1759-1969` → `routes/users.py`
- (etc. — see structure above)

**Things to be careful about:**
- The OpenAI clients (`openai_client`, `async_openai_client`) and the FAISS
  index are module-level globals in `main.py`. They need to move to a shared
  `dependencies.py` or `clients.py` module so route files can import them
  without re-initializing.
- Tests that import from `main` (e.g., `from main import app`) will need to
  keep working — `main.py` should still expose `app`.

---

## 2. Wrap remaining blocking I/O in `asyncio.to_thread()`

Phase 2 fixed the biggest offender (OpenAI calls in `/chat`, `/chat/stream`,
`/chat/voice`) by switching to `AsyncOpenAI`. Still synchronous and blocking
the event loop:

- **Firestore reads/writes** in `firebase_service.py` (`doc_ref.get()`,
  `.set()`, `.update()`, `.stream()`). Each call is ~50–200ms but inside an
  async handler, that's 50–200ms of frozen event loop.
- **LangChain embeddings** (`embeddings_model.embed_query()`) called from
  RAG retrieval at `main.py:342`.
- **Pickle deserialization** of the FAISS index at startup (`main.py:305`).

**Fix pattern:**
```python
import asyncio

# Before (blocks event loop):
result = doc_ref.get()

# After (runs in FastAPI's thread pool):
result = await asyncio.to_thread(doc_ref.get)
```

Apply systematically to every Firestore call site inside an `async def` handler.

---

## 3. PII redaction in logs

`main.py` and several services log `user_id` directly:

```python
logger.info(f"👤 Loaded profile for user {request.user_id}")
```

For HIPAA-style compliance, replace with hashed or truncated IDs in logs:

```python
def _redact_uid(uid: str) -> str:
    return f"{uid[:4]}...{uid[-2:]}" if len(uid) > 8 else "***"

logger.info(f"Loaded profile for user {_redact_uid(request.user_id)}")
```

Add a helper to `utils/logging.py` and use it everywhere user IDs / emails
appear in log output.

---

## 4. Replace `print()` calls with `logger`

`firebase_service.py` and a few other files use `print(...)` with emojis for
status output. These bypass the structured JSON formatter, so they show up as
unparseable text in Cloud Logging. Replace every `print()` in non-test code
with `logger.info() / .warning() / .error()`.

---

## 5. Strict dependency pinning

`requirements.txt` uses loose version ranges (`langchain>=0.2.14`). This
means `pip install` can pull a breaking new version on a fresh build.

**Fix:** generate a `requirements.lock` with `pip-compile` (from `pip-tools`)
or switch to `poetry`. CI should install from the lockfile, not the loose file.

---

## 6. Non-root Docker user

Dockerfile runs the app as root. Add before `CMD`:

```dockerfile
RUN useradd -m -u 1000 lucille && chown -R lucille:lucille /app
USER lucille
```

Also add a `.dockerignore` excluding `venv/`, `__pycache__`, `.git`, `tests/`,
`*.pyc`, `.env`, `node_modules`.

---

## 7. OpenTelemetry distributed tracing

Phase 3 added per-request correlation IDs via `X-Request-ID` (logs are
already grouped). The next step is full OpenTelemetry instrumentation:

- Trace a request across FastAPI → OpenAI → Firestore → response
- Visualize in Cloud Trace / Jaeger
- Auto-instrument FastAPI with `opentelemetry-instrumentation-fastapi`

---

## 8. Remove the `escape-ujuzxr` stale reference

`firebase_service.py:68` references the old project ID `escape-ujuzxr` in an
error message. Update to the current project ID (`escape-self-care-ai`) or
just remove the project-specific URL.

---

## 9. Remove the hardcoded fallback service-account filename

`firebase_service.py:41` falls back to `"escape-self-care-ai-firebase-key.json"`
if `FIREBASE_CREDENTIAL_PATH` isn't set. This works today only because the
filename happens to match the project. Better: require the env var explicitly,
fail loudly if missing in production.

---

## 10. Firebase Emulator setup for local development

Add an `emulators` section to `firebase.json` and document running:

```bash
firebase emulators:start --only firestore,auth
```

So local development can iterate against an emulated Firebase without
touching the real project. Set `FIRESTORE_EMULATOR_HOST=localhost:8080`
in `.env.local` and have `firebase_service.py` detect and use it.
