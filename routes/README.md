# Router Migration Guide

## Status
The `routes/` directory is set up and ready for incremental endpoint migration from `main.py`.

## Why Incremental?
`main.py` (4400+ lines) has deep coupling between endpoints and module-level state:
- `VectorStore`, `DOCS` (FAISS index) — used by /chat, /health
- `OPENAI_MODEL`, `client` — used by /chat, /health
- `session_store` — used by /chat
- Helper functions (`validate_session_id`, etc.) — used by /chat, /users

Moving endpoints to routers requires passing these via FastAPI dependency injection or a shared state module.

## Migration Order (safest first)

### Phase 1: No local state dependencies (safe to move now)
- `routes/assessments.py` — /assessments/* (8 endpoints)
- `routes/voice.py` — /tts, /stt (2 endpoints)
- `routes/wearables.py` — /wearables/* (4 endpoints)
- `routes/reviews.py` — /reviews/* (3 endpoints)
- `routes/rl.py` — /rl/* (1 endpoint)

### Phase 2: Service-only dependencies
- `routes/therapy.py` — /therapy/* (13 endpoints)
- `routes/feedback.py` — /feedback/* (4 endpoints)
- `routes/soundscapes.py` — /soundscapes/* (8 endpoints)
- `routes/safety.py` — /safety/* (3 endpoints)
- `routes/users.py` — /users/* (16 endpoints)
- `routes/finetuning.py` — /finetuning/* (6 endpoints)
- `routes/admin.py` — /admin/* (16 endpoints)

### Phase 3: Heavy local state (move last)
- `routes/chat.py` — /chat/* (6 endpoints, ~1100 lines)
  - Requires refactoring VectorStore, FAISS, session_store into a shared module
- `routes/health.py` — /health/* (4 endpoints)
  - References OPENAI_MODEL, VectorStore, DOCS

## How to Migrate an Endpoint

```python
# routes/voice.py
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import logging

from voice_service import get_voice_service
from config import get_config
from models import TTSRequest, TTSResponse, STTRequest, STTResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])

@router.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    # ... move endpoint body here ...

# main.py
from routes.voice import router as voice_router
app.include_router(voice_router)
# Then delete the original endpoint from main.py
```
