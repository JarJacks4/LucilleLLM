"""
LucilleLLM - Self-Care Chatbot API

A FastAPI-based chatbot that provides self-care advice and wellbeing support.
Features OpenAI integration, FAISS vector search, and Firebase session management.

RAG Enhancement (suyash-rag-enhancement branch):
- Added RAG (Retrieval-Augmented Generation) to /chat endpoint
- Retrieves relevant context from FAISS vector store based on user queries
- Augments LLM prompts with retrieved self-care knowledge base content
- Improves response accuracy and relevance for self-care topics

Streaming Enhancement (suyash-streaming-chat branch):
- Added /chat/stream endpoint for real-time token streaming
- Uses Server-Sent Events (SSE) for true streaming responses
- Provides ChatGPT-like typing experience with RAG integration
"""

from fastapi.exceptions import RequestValidationError
from fastapi import Depends, FastAPI, Request, HTTPException, status, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Optional, AsyncGenerator
from dotenv import load_dotenv
import os
import uuid
import json
import asyncio
from collections import defaultdict
from datetime import datetime
import logging
import sys
import numpy as np
import pickle

# OpenAI and LangChain imports
from openai import OpenAI, AsyncOpenAI, APIConnectionError, RateLimitError
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
import tiktoken
import httpx

# Local imports
from firebase_service import get_firebase_service
from user_service import get_user_service
from emotion_service import get_emotion_service
from memory_service import get_memory_service
from prompt_engine import get_prompt_engine, PromptContext
from therapy_service import get_therapy_service
from progress_service import get_progress_service
from feedback_service import get_feedback_service
from soundscape_service import get_soundscape_service, EXERCISE_SOUNDSCAPE_MAP
from safety_service import get_safety_service
from chat_agent_service import get_chat_agent_service
from dependency_service import get_dependency_service
from cultural_service import get_cultural_service
from compliance_service import get_compliance_service
from audit_service import get_audit_service
from storage_service import get_storage_service
from voice_service import get_voice_service
from wearable_service import get_wearable_service
from rl_service import get_rl_service
from finetuning_service import get_finetuning_service
from monitoring_service import get_monitoring_service
from escalation_service import get_escalation_service
from assessment_service import get_assessment_service
from config import get_config
from cache import get_cache, user_profile_key, effectiveness_key
from middleware import RateLimitMiddleware, MetricsMiddleware, PrivacyMiddleware, get_metrics_collector, get_request_id
from auth_middleware import get_current_user, require_admin, require_same_user, _is_admin
from utils.sanitize import sanitize_input
from models import (
    ChatRequest, ChatResponse,
    OnboardingRequest, OnboardingResponse,
    UserProfileResponse, UserProfileUpdateRequest,
    UserProfile, PersonaLayer, AffectiveLayer, BehavioralLayer,
    MotivationalLayer, CognitiveLayer, Goal, MoodEntry,
    EmotionDetectionResult,
    Memory, MemoryType, MemoryCreateRequest, MemorySearchRequest,
    StartExerciseRequest, StartExerciseResponse, TherapyModality,
    PracticeTask, TaskStatus, CreateTaskRequest, UpdateTaskRequest,
    ProgressSummary,
    FeedbackRating, ResponseFeedback, ExerciseOutcome,
    SubmitFeedbackRequest, SubmitOutcomeRequest, EffectivenessProfile,
    SoundscapeCategory, SoundscapeTemplate, SoundscapeSession,
    SoundscapeRecommendation, StartSoundscapeRequest,
    RiskLevel, SafetyCheckResult, SafetyEvent, SafetyEventType,
    CrisisResource, SafetyCheckRequest,
    DependencyRiskLevel,
    ConsentRequest, ConsentRecord,
    VoiceChatRequest, VoiceChatResponse,
    TTSRequest, TTSResponse,
    STTRequest, STTResponse,
    SleepRecord, ActivityRecord, DailyHealthMetrics,
    HealthSyncRequest, HealthSummary,
    ExtractTrainingDataRequest, SubmitFineTuningJobRequest,
    FineTuningStatsResponse,
    DashboardSystemMetrics, DashboardEngagementMetrics,
    DashboardTherapyMetrics, DashboardSafetyMetrics,
    DashboardModelMetrics, DashboardRLMetrics, DashboardAllMetrics,
    EscalationEvent, UpdateEscalationRequest, EscalationStats,
    GenerateReviewRequest, AnnualReview,
    AssessmentType, StartAssessmentRequest, StartAssessmentResponse,
    SubmitAnswerRequest, SubmitAnswerResponse,
    CompleteAssessmentResponse, AssessmentHistoryResponse, WellnessScore,
)


# Load environment variables
load_dotenv()

# Configure structured logging for production (Cloud Run / Cloud Logging compatible)


class StructuredFormatter(logging.Formatter):
    """JSON log formatter compatible with Google Cloud Logging.

    Automatically includes the per-request correlation ID (X-Request-ID) on
    every log line, so you can grep all logs from a single request even when
    they come from different modules and concurrent requests are interleaved.
    """

    def format(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        # Include request correlation ID when inside a request context
        rid = get_request_id()
        if rid:
            log_entry["request_id"] = rid
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


_log_config = get_config()
_log_handler = logging.StreamHandler(sys.stdout)
if _log_config.LOG_FORMAT == "json":
    _log_handler.setFormatter(StructuredFormatter())
else:
    _log_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.basicConfig(
    level=getattr(logging, _log_config.LOG_LEVEL, logging.INFO),
    handlers=[_log_handler],
)
logger = logging.getLogger(__name__)


def get_openai_api_key():
    """Get OpenAI API key from environment or Google Secret Manager"""
    # Try local environment first
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key.strip()

    # Try Google Secret Manager in production
    try:
        from google.cloud import secretmanager
        # import google.auth

        client = secretmanager.SecretManagerServiceClient()
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')

        # If no project ID set, try to infer it
        if not project_id:
            try:
                creds, inferred_project = google.auth.default()
                if inferred_project:
                    project_id = inferred_project
            except Exception:
                pass

        if project_id:
            secret_name = f"projects/{project_id}/secrets/openai-api-key/versions/latest"
            response = client.access_secret_version(
                request={"name": secret_name})
            return response.payload.data.decode("UTF-8").strip()
    except Exception as e:
        print(f"Warning: Could not load from Secret Manager: {e}")

    return None


# Initialize OpenAI API key
openai_api_key = get_openai_api_key()
if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not set. Please set it in environment or Google Secret Manager.")

os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize OpenAI clients with proper timeout configuration.
# We keep BOTH a sync and an async client:
#   - openai_client (sync): used by services (emotion, memory, finetuning, etc.)
#     that take a client by injection and were written for the sync API.
#   - async_openai_client: used by FastAPI async handlers (/chat, /chat/stream,
#     /chat/voice) so OpenAI calls don't block the event loop. Switching async
#     handlers to AsyncOpenAI is the single biggest perf win — under load, the
#     event loop can serve other requests during the 2-30s LLM round-trip
#     instead of stalling.
_httpx_client = httpx.Client(
    timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0),
)
_httpx_async_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0),
)

openai_client = OpenAI(
    api_key=openai_api_key,
    http_client=_httpx_client,
)
async_openai_client = AsyncOpenAI(
    api_key=openai_api_key,
    http_client=_httpx_async_client,
)

# Configuration (from environment variables via config service)
_app_config = get_config()
OPENAI_MODEL = _app_config.OPENAI_MODEL
OPENAI_VISION_MODEL = _app_config.OPENAI_VISION_MODEL
EMBEDDING_MODEL = _app_config.EMBEDDING_MODEL
FAISS_INDEX_PATH = _app_config.FAISS_INDEX_PATH

# Vision client for image-based mood analysis. Uses a dedicated key if
# OPENAI_VISION_API_KEY is set, otherwise reuses the main OpenAI key/client.
_vision_api_key = os.getenv("OPENAI_VISION_API_KEY", "").strip()
if _vision_api_key and _vision_api_key != openai_api_key:
    vision_openai_client = OpenAI(
        api_key=_vision_api_key, http_client=_httpx_client)
else:
    vision_openai_client = openai_client

# Initialize emotion detection service (text via main client, images via vision client)
get_emotion_service(
    openai_client=openai_client,
    model=OPENAI_MODEL,
    vision_client=vision_openai_client,
    vision_model=OPENAI_VISION_MODEL,
)

# Initialize memory service (uses same OpenAI client + embedding model)
get_memory_service(openai_client=openai_client,
                   embedding_model=EMBEDDING_MODEL)

# Inject OpenAI client into fine-tuning service (Phase 18)
try:
    _ft_svc = get_finetuning_service()
    _ft_svc.set_openai_client(openai_client)
except Exception as _ft_err:
    print(
        f"Warning: Could not inject OpenAI client into FT service: {_ft_err}")

# Inject OpenAI client into escalation service (Phase 20)
try:
    _esc_svc = get_escalation_service()
    _esc_svc.set_openai_client(openai_client)
except Exception as _esc_err:
    print(
        f"Warning: Could not inject OpenAI client into escalation service: {_esc_err}")

# Initialize FastAPI app
app = FastAPI(
    title="LucilleLLM API",
    description="Self-care chatbot API providing wellbeing support and advice",
    version="1.0.0"
)

# CORS middleware
# Locked to local dev origins. Mobile apps don't enforce CORS, so they're
# unaffected. When you deploy a production web frontend, add its origin to
# CORS_ALLOWED_ORIGINS in your environment (comma-separated), e.g.
#   CORS_ALLOWED_ORIGINS=https://app.lucille.com,https://lucille.com
_DEFAULT_DEV_ORIGINS = [
    "http://localhost:3000",   # Next.js / Create React App
    "http://localhost:5173",   # Vite
    "http://localhost:8080",   # generic dev / FastAPI test UI
    "http://localhost:4200",   # Angular
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
]
_extra_origins = [
    o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_DEV_ORIGINS + _extra_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Rate limiting, metrics, and privacy middleware
# Starlette order: last added = outermost.
# PrivacyMiddleware -> MetricsMiddleware -> RateLimitMiddleware (innermost)
app.add_middleware(PrivacyMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(RateLimitMiddleware)


# ── Startup Event ──────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Log service initialization status on startup."""
    import time as _time
    start = _time.time()
    logger.info("Starting LucilleLLM services...")
    services = {
        "firebase": get_firebase_service,
        "user": get_user_service,
        "emotion": lambda: get_emotion_service(),
        "memory": lambda: get_memory_service(),
        "therapy": get_therapy_service,
        "progress": get_progress_service,
        "feedback": get_feedback_service,
        "soundscape": get_soundscape_service,
        "safety": get_safety_service,
        "prompt_engine": get_prompt_engine,
        "chat_agent": get_chat_agent_service,
        "dependency": get_dependency_service,
        "cultural": get_cultural_service,
        "compliance": get_compliance_service,
        "audit": get_audit_service,
        "storage": get_storage_service,
        "voice": get_voice_service,
        "wearable": get_wearable_service,
        "rl": get_rl_service,
        "finetuning": get_finetuning_service,
        "monitoring": get_monitoring_service,
        "escalation": get_escalation_service,
    }
    for name, getter in services.items():
        try:
            getter()
            logger.info(f"Service initialized: {name}")
        except Exception as e:
            logger.error(f"Service failed to initialize: {name} — {e}")
    elapsed = _time.time() - start
    logger.info(f"Startup complete in {elapsed:.2f}s")


# Initialize embeddings and vector store
embeddings_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=openai_api_key,
)


def load_faiss_vectorstore(local_path: str) -> FAISS:
    """Load FAISS index from disk"""
    os.makedirs(local_path, exist_ok=True)
    vs = FAISS.load_local(local_path, embeddings=embeddings_model,
                          allow_dangerous_deserialization=True)
    print(f"✅ Loaded FAISS vector store (dimension: {vs.index.d})")
    return vs


# Load vector store
VectorStore = load_faiss_vectorstore(FAISS_INDEX_PATH)

# Load document texts for RAG
try:
    with open('texts.pkl', 'rb') as file:
        DOCS = pickle.load(file)
    logger.info(f"✅ Loaded {len(DOCS)} documents for RAG")
except FileNotFoundError:
    logger.warning("⚠️ texts.pkl not found. RAG retrieval will be disabled.")
    DOCS = []
except Exception as e:
    logger.error(f"❌ Failed to load texts.pkl: {e}")
    DOCS = []

# Initialize tokenizer for token counting
try:
    tokenizer = tiktoken.encoding_for_model(OPENAI_MODEL)
except KeyError:
    tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text"""
    return len(tokenizer.encode(text))


def retrieve_relevant_context(query: str, k: int = 5, similarity_threshold: float = 1.1) -> str:
    """
    Retrieve relevant context from FAISS vector store for RAG.

    Args:
        query: User's query
        k: Number of top results to retrieve
        similarity_threshold: Maximum distance threshold (lower is more similar)

    Returns:
        Combined context string from retrieved documents
    """
    if not DOCS or len(DOCS) == 0:
        logger.warning("No documents available for RAG retrieval")
        return ""

    try:
        # Embed the query using OpenAI embeddings
        query_embedding = embeddings_model.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)

        # Search in FAISS index
        distances, indices = VectorStore.index.search(query_vector, k)

        # Filter by similarity threshold and collect relevant docs
        retrieved_contexts = []
        for idx, distance in zip(indices[0], distances[0]):
            if distance <= similarity_threshold and 0 <= idx < len(DOCS):
                retrieved_contexts.append(DOCS[idx].page_content)

        if retrieved_contexts:
            logger.info(
                f"🔍 Retrieved {len(retrieved_contexts)} relevant documents for query")
            return "\n\n".join(retrieved_contexts)
        else:
            logger.info(
                f"No documents found within similarity threshold ({similarity_threshold})")
            return ""

    except Exception as e:
        logger.error(f"❌ Error retrieving context: {e}")
        return ""


# Session management
session_histories: Dict[str, BaseChatMessageHistory] = defaultdict(
    ChatMessageHistory)
session_summaries: Dict[str, str] = defaultdict(str)

# Initialize summarization LLM
summary_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0,
                         request_timeout=120, max_retries=3)

# Session validation function


def validate_session_id(session_id: str) -> str:
    """Validate and normalize session ID"""
    if not session_id or session_id == "unique_identifier" or session_id == "null" or session_id == "":
        return str(uuid.uuid4())
    return session_id

# Request/Response models now imported from models.py

# Main chat endpoint


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: dict = Depends(get_current_user)):
    """Main chat endpoint with Firebase session management.

    Authentication: requires either a Firebase ID token or the
    INTERNAL_SERVICE_KEY (for backend/test calls). Authenticated users may
    only chat as themselves; admins and the service account may chat as any
    user_id (useful for testing and backfills).
    """
    try:
        # Enforce that the request's user_id matches the authenticated user.
        # Admins and the service account can chat as any user.
        if not _is_admin(user):
            if request.user_id and request.user_id != user.get("uid"):
                raise HTTPException(
                    status_code=403,
                    detail="user_id in request does not match authenticated user",
                )
            # Default user_id to the authenticated uid if not provided
            if not request.user_id:
                request.user_id = user.get("uid")

        # Validate and normalize session ID
        session_id = validate_session_id(request.session_id)
        prompt = sanitize_input(request.message)

        # Initialize Firebase service
        firebase_service = get_firebase_service()

        # Load user profile if user_id is provided (with cache)
        user_profile_text = ""
        comm_pref = "empathetic"
        profile_data = None
        if request.user_id:
            try:
                user_service = get_user_service()
                cache = get_cache()
                cache_key = user_profile_key(request.user_id)
                profile_data = cache.get(cache_key)
                if profile_data is None:
                    profile_data = user_service.get_user_profile(
                        request.user_id)
                    if profile_data:
                        cache.set(cache_key, profile_data,
                                  ttl=get_config().CACHE_TTL_USER_PROFILE)
                if profile_data:
                    user_profile_text = user_service.format_profile_for_prompt(
                        profile_data)
                    comm_pref = (
                        profile_data.get("persona", {})
                        .get("communication_preference", "empathetic")
                    )
                    logger.info(f"👤 Loaded profile for user {request.user_id}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load user profile: {e}")

        # ── Phase 12: Cultural Context Extraction ──
        cultural_context_text = ""
        country_code = "US"
        if request.user_id and profile_data:
            try:
                cultural_svc = get_cultural_service()
                cultural_ctx = cultural_svc.extract_cultural_context(
                    profile_data)
                country_code = cultural_ctx.country_code
                cultural_context_text = cultural_svc.get_cultural_prompt_context(
                    cultural_ctx
                )
            except Exception as e:
                logger.warning(f"Cultural context extraction failed: {e}")

        # ── Phase 16: Health Context from Wearables ──
        health_context_text = ""
        if request.user_id:
            try:
                wearable_svc = get_wearable_service()
                health_context_text = wearable_svc.format_health_context(
                    request.user_id)
                if health_context_text:
                    logger.info(
                        f"💓 Health context loaded for user {request.user_id}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load health context: {e}")

        # Detect emotion and intent from user message
        detection_result = None
        if request.user_id:
            try:
                emotion_svc = get_emotion_service()
                detection_result = emotion_svc.detect(prompt)

                # Auto-store mood entry in user's affective layer
                mood_entry = {
                    "mood": detection_result.emotion,
                    "intensity": detection_result.intensity,
                    "context": prompt[:200],
                    "recorded_at": datetime.now().isoformat(),
                    "detected_via": "text_auto",
                    "confidence": detection_result.confidence,
                    "intent": detection_result.intent,
                }
                user_service = get_user_service()
                user_service.append_mood_entry(request.user_id, mood_entry)
            except Exception as e:
                logger.warning(f"⚠️ Emotion detection failed, continuing: {e}")
                detection_result = None

        # ── Phase 9: Input Safety Screening (fast, no API call) ──
        safety_check = SafetyCheckResult()
        safety_svc = get_safety_service()
        try:
            safety_check = safety_svc.check_input(prompt)
            if safety_check.risk_level != RiskLevel.LOW:
                logger.warning(
                    f"Safety check: risk={safety_check.risk_level.value}, "
                    f"flags={safety_check.flags}"
                )
        except Exception as e:
            logger.warning(f"Safety screening failed, continuing: {e}")

        # If CRITICAL risk, bypass LLM and return crisis response immediately
        if safety_check.risk_level == RiskLevel.CRITICAL:
            crisis_response = safety_svc.get_crisis_intercept_response()

            # Log the safety event
            if request.user_id:
                try:
                    safety_svc.log_safety_event(
                        user_id=request.user_id,
                        session_id=session_id,
                        event_type=SafetyEventType.CRITICAL_INTERCEPT,
                        risk_level=RiskLevel.CRITICAL,
                        message_snippet=prompt[:200],
                        flags=safety_check.flags,
                        action_taken="crisis_intercept",
                    )
                except Exception:
                    pass

                # Phase 20: Auto-escalation for CRITICAL safety
                try:
                    esc_svc = get_escalation_service()
                    esc_svc.check_and_create_escalation(
                        user_id=request.user_id,
                        safety_risk_level=RiskLevel.CRITICAL.value,
                        safety_event_id=None,
                        dependency_risk_level=DependencyRiskLevel.NONE.value,
                        dependency_score=0,
                    )
                except Exception:
                    pass

            # Update chat history for continuity
            chat_history = session_histories.get(
                session_id, ChatMessageHistory())
            chat_history.add_user_message(prompt)
            chat_history.add_ai_message(crisis_response)
            session_histories[session_id] = chat_history

            payload = ChatResponse(
                session_id=session_id,
                response=crisis_response,
                conversation=[prompt, crisis_response],
                status="success",
                timestamp=datetime.now().isoformat(),
                message_count=len(chat_history.messages),
                user_id=request.user_id,
                detected_emotion=detection_result.emotion if detection_result else None,
                detected_intent="crisis",
            )
            return JSONResponse(content=payload.model_dump())

        # ── Phase 12: Dependency Detection (fast, no extra API call) ──
        dependency_override_text = ""
        if request.user_id:
            try:
                dep_svc = get_dependency_service()
                dep_metrics = dep_svc.record_interaction(
                    request.user_id, prompt)
                dep_assessment = dep_svc.assess_dependency(
                    request.user_id, dep_metrics, prompt
                )
                if dep_assessment.risk_level != DependencyRiskLevel.NONE:
                    dependency_override_text = dep_svc.format_dependency_for_prompt(
                        dep_assessment
                    )
                    logger.info(
                        f"Dependency: risk={dep_assessment.risk_level.value}, "
                        f"score={dep_assessment.score}, signals={dep_assessment.signals}"
                    )
                elif dep_svc.should_show_self_care_reminder(dep_metrics):
                    dependency_override_text = (
                        "SELF-CARE REMINDER: Subtly include a brief self-care check-in "
                        "in your response (e.g., 'Have you taken a break today?'). "
                        "Keep it natural and brief."
                    )
            except Exception as e:
                logger.warning(f"Dependency check failed, continuing: {e}")

        # Retrieve relevant memories for context
        memory_context_text = ""
        if request.user_id:
            try:
                memory_svc = get_memory_service()
                relevant_memories = memory_svc.search_memories(
                    request.user_id, prompt, limit=5
                )
                memory_context_text = memory_svc.format_memories_for_prompt(
                    relevant_memories
                )
            except Exception as e:
                logger.warning(f"⚠️ Memory retrieval failed, continuing: {e}")

        # Load existing session from Firebase if it exists
        try:
            existing_session = firebase_service.get_chat_session(session_id)
            if existing_session:
                summary = existing_session.get('summary', '')
                session_summaries[session_id] = summary
                print(f"📱 Loaded existing session {session_id} from Firebase")
            else:
                summary = session_summaries.get(session_id, "")
        except Exception as e:
            print(f"⚠️ Failed to load session from Firebase: {e}")
            summary = session_summaries.get(session_id, "")
            existing_session = None

        # 🔍 RAG: Retrieve relevant context from vector store
        retrieved_context = retrieve_relevant_context(
            prompt, k=5, similarity_threshold=1.1)

        # Build emotion context text for prompt engine
        emotion_context_text = ""
        if detection_result:
            emotion_context_text = get_emotion_service().format_detection_for_prompt(
                detection_result
            )

        # Check for active therapy exercise and due tasks
        active_exercise_text = ""
        due_tasks_text = ""
        if request.user_id:
            try:
                therapy_svc = get_therapy_service()
                active_ex = therapy_svc.get_active_exercise(request.user_id)
                if active_ex:
                    active_exercise_text = therapy_svc.format_exercise_for_prompt(
                        active_ex)
            except Exception as e:
                logger.warning(f"Failed to load active exercise: {e}")

            # Load due/overdue practice tasks for prompt context
            try:
                therapy_svc = get_therapy_service()
                due_tasks = therapy_svc.get_due_tasks(request.user_id)
                if due_tasks:
                    due_tasks_text = therapy_svc.format_due_tasks_for_prompt(
                        due_tasks)
            except Exception as e:
                logger.warning(f"Failed to load due tasks: {e}")

        # Load feedback effectiveness insights for prompt
        feedback_insights_text = ""
        if request.user_id:
            try:
                feedback_svc = get_feedback_service()
                effectiveness = feedback_svc.compute_effectiveness(
                    request.user_id
                )
                if effectiveness.total_outcomes >= 2:
                    feedback_insights_text = (
                        feedback_svc.format_insights_for_prompt(effectiveness)
                    )
            except Exception as e:
                logger.warning(f"Failed to load feedback insights: {e}")

        # Load soundscape suggestion for prompt (when emotion intensity >= 5)
        soundscape_suggestion_text = ""
        if detection_result and detection_result.intensity >= 5:
            try:
                ss_svc = get_soundscape_service()
                soundscape_suggestion_text = ss_svc.format_suggestion_for_prompt(
                    detection_result.emotion, detection_result.intensity
                )
            except Exception as e:
                logger.warning(f"Failed to load soundscape suggestion: {e}")

        # Build safety override text for prompt engine
        safety_override_text = ""
        if safety_check.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) \
                or safety_check.jailbreak_detected:
            safety_override_text = safety_svc.get_safety_prompt_override(
                safety_check
            )

        # Assemble system prompt via prompt engine
        prompt_ctx = PromptContext(
            user_profile_text=user_profile_text,
            memory_context_text=memory_context_text,
            conversation_summary=summary,
            retrieved_rag_context=retrieved_context,
            detection_result=detection_result,
            emotion_context_text=emotion_context_text,
            communication_preference=comm_pref,
            active_exercise_text=active_exercise_text,
            due_tasks_text=due_tasks_text,
            feedback_insights_text=feedback_insights_text,
            soundscape_suggestion_text=soundscape_suggestion_text,
            dependency_override_text=dependency_override_text,
            cultural_context_text=cultural_context_text,
            health_context_text=health_context_text,
            safety_override_text=safety_override_text,
        )
        system_prompt = get_prompt_engine().build_system_prompt(prompt_ctx)

        # Get chat history for context
        chat_history = session_histories.get(session_id, ChatMessageHistory())
        history_messages = []
        for msg in chat_history.messages[-6:]:  # Last 6 messages for context
            if isinstance(msg, HumanMessage):
                history_messages.append(
                    {"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history_messages.append(
                    {"role": "assistant", "content": msg.content})

        # Create messages for OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # Phase 18: Resolve model for A/B routing
        model_to_use = OPENAI_MODEL
        is_fine_tuned = False
        ab_group = "base"
        try:
            ft_svc = get_finetuning_service()
            model_to_use, is_fine_tuned, ab_group = ft_svc.resolve_model_for_user(
                request.user_id
            )
            if is_fine_tuned:
                logger.info(
                    f"FT A/B: user {request.user_id} -> fine-tuned model")
        except Exception:
            model_to_use = OPENAI_MODEL

        # Call OpenAI API with tool calling
        try:
            agent_svc = get_chat_agent_service()
            tool_defs = agent_svc.get_tool_definitions()

            try:
                response = await async_openai_client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7,
                    tools=tool_defs,
                    tool_choice="auto",
                )
            except Exception as model_err:
                if is_fine_tuned:
                    logger.warning(
                        f"Fine-tuned model failed, falling back to base: {model_err}")
                    model_to_use = OPENAI_MODEL
                    is_fine_tuned = False
                    ab_group = "base"
                    response = await async_openai_client.chat.completions.create(
                        model=model_to_use,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7,
                        tools=tool_defs,
                        tool_choice="auto",
                    )
                else:
                    raise

            # Tool-calling loop (max 5 iterations)
            for _iteration in range(5):
                response_message = response.choices[0].message
                if not response_message.tool_calls:
                    break
                messages.append(response_message)
                for tool_call in response_message.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}
                    logger.info(f"Tool call: {fn_name}({fn_args})")
                    tool_result = agent_svc.execute_tool_call(
                        tool_name=fn_name,
                        tool_args=fn_args,
                        user_id=request.user_id,
                        retrieve_context_fn=retrieve_relevant_context,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    })
                response = await async_openai_client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7,
                    tools=tool_defs,
                    tool_choice="auto",
                )

            bot_response = response.choices[0].message.content or ""
        except APIConnectionError as e:
            print(f"❌ OpenAI network error: {e}")
            raise HTTPException(
                status_code=502, detail=f"OpenAI network error: {e}")
        except RateLimitError as e:
            print(f"❌ OpenAI rate limit: {e}")
            raise HTTPException(
                status_code=429, detail="OpenAI rate limit hit")
        except Exception as e:
            print(f"❌ OpenAI API call failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"OpenAI API error: {e}")

        # ── Phase 9: Output Safety Validation ──
        try:
            bot_response, output_modified = safety_svc.validate_output(
                bot_response, safety_check, country_code=country_code
            )
            if output_modified:
                logger.info(
                    "Safety: bot response was modified by output validation")
        except Exception as e:
            logger.warning(f"Output validation failed, using original: {e}")

        # Log safety events for non-low risk
        safety_event_id = None
        if request.user_id and safety_check.risk_level != RiskLevel.LOW:
            try:
                event_type = SafetyEventType.CRISIS_DETECTED if safety_check.crisis_detected \
                    else SafetyEventType.HIGH_RISK_INPUT
                if safety_check.jailbreak_detected:
                    event_type = SafetyEventType.JAILBREAK_ATTEMPT
                if output_modified:
                    event_type = SafetyEventType.OUTPUT_MODIFIED
                safety_event_id = safety_svc.log_safety_event(
                    user_id=request.user_id,
                    session_id=session_id,
                    event_type=event_type,
                    risk_level=safety_check.risk_level,
                    message_snippet=prompt[:200],
                    flags=safety_check.flags,
                    action_taken=safety_check.action_taken,
                )
            except Exception:
                pass

            # Phase 20: Auto-escalation check for non-LOW safety events
            try:
                _dep_risk = DependencyRiskLevel.NONE.value
                _dep_score = 0
                if 'dep_assessment' in dir():
                    _dep_risk = dep_assessment.risk_level.value
                    _dep_score = dep_assessment.score
                esc_svc = get_escalation_service()
                esc_svc.check_and_create_escalation(
                    user_id=request.user_id,
                    safety_risk_level=safety_check.risk_level.value,
                    safety_event_id=safety_event_id,
                    dependency_risk_level=_dep_risk,
                    dependency_score=_dep_score,
                )
            except Exception:
                pass

        # Update chat history
        chat_history.add_user_message(prompt)
        chat_history.add_ai_message(bot_response)
        session_histories[session_id] = chat_history

        # Reconstruct conversation for response
        conversation_strings = [prompt, bot_response]

        # Summarize if over token limit
        full_text = f"{prompt} {bot_response}"
        if count_tokens(full_text) > 8000:
            print("🔁 Summarizing chat history...")
            try:
                # Create a simple summary using the LLM directly
                summary_prompt = f"Please summarize the following conversation history in a concise way, focusing on key topics and decisions:\n\n{full_text}"
                refined_summary = summary_llm.invoke(summary_prompt).content
                session_summaries[session_id] = refined_summary

                # Update summary in Firebase
                firebase_service.update_session_summary(
                    session_id, refined_summary)
            except Exception as e:
                print(f"⚠️ Summarization failed: {e}")

        # Store/Update session in Firebase
        try:
            current_messages = chat_history.messages
            current_summary = session_summaries.get(session_id, "")

            if existing_session:
                firebase_service.update_chat_session(
                    session_id, current_messages, current_summary)
            else:
                firebase_service.store_chat_session(
                    session_id, current_messages, current_summary)
        except Exception as e:
            print(f"⚠️ Failed to store session in Firebase: {e}")

        # Link session to user if user_id was provided
        if request.user_id:
            try:
                firebase_service.link_session_to_user(
                    session_id, request.user_id)
            except Exception as e:
                logger.warning(f"⚠️ Failed to link session to user: {e}")

        # Extract and store memories from this conversation turn
        if request.user_id:
            try:
                memory_svc = get_memory_service()
                memory_svc.extract_memories(
                    request.user_id, prompt, bot_response)
            except Exception as e:
                logger.warning(f"⚠️ Memory extraction failed: {e}")

        # Phase 18: Log model performance for A/B tracking
        try:
            ft_svc = get_finetuning_service()
            if ft_svc.is_available:
                ft_svc.log_model_performance(
                    user_id=request.user_id or "",
                    session_id=session_id,
                    model_used=model_to_use,
                    is_fine_tuned=is_fine_tuned,
                    ab_group=ab_group,
                    response_length=len(bot_response),
                    detected_emotion=detection_result.emotion if detection_result else "",
                    detected_intent=detection_result.intent if detection_result else "",
                )
        except Exception as e:
            logger.warning(f"FT performance logging failed: {e}")

        payload = ChatResponse(
            session_id=session_id,
            response=bot_response,
            conversation=conversation_strings,
            status="success",
            timestamp=datetime.now().isoformat(),
            message_count=len(chat_history.messages),
            user_id=request.user_id,
            detected_emotion=detection_result.emotion if detection_result else None,
            detected_intent=detection_result.intent if detection_result else None,
            model_used=model_to_use,
        )
        return JSONResponse(content=payload.model_dump())

    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming chat endpoint


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, user: dict = Depends(get_current_user)):
    """
    Streaming chat endpoint with RAG enhancement (FlutterFlow-compatible).
    Returns Server-Sent Events (SSE) for real-time token streaming.

    Auth: same as /chat — Firebase ID token or INTERNAL_SERVICE_KEY.

    ALL messages have the SAME format:
    {
        "content": "token",      <- The streamed token (empty string when done)
        "done": false/true,      <- false during streaming, true when complete
        "session_id": "...",     <- Session ID
        "response": "...",       <- Empty during streaming, full response when done
        "message_count": N       <- 0 during streaming, actual count when done
    }

    Final signal: data: [DONE]
    """
    try:
        # Enforce user_id matches authenticated user (admin/service account exempt)
        if not _is_admin(user):
            if request.user_id and request.user_id != user.get("uid"):
                raise HTTPException(
                    status_code=403,
                    detail="user_id in request does not match authenticated user",
                )
            if not request.user_id:
                request.user_id = user.get("uid")

        # Validate and normalize session ID
        session_id = validate_session_id(request.session_id)
        prompt = sanitize_input(request.message)

        logger.info(f"Streaming chat request for session {session_id}")

        # Initialize Firebase service
        firebase_service = get_firebase_service()

        # Load user profile if user_id is provided (with cache)
        user_profile_text = ""
        comm_pref = "empathetic"
        profile_data = None
        if request.user_id:
            try:
                user_service = get_user_service()
                cache = get_cache()
                cache_key = user_profile_key(request.user_id)
                profile_data = cache.get(cache_key)
                if profile_data is None:
                    profile_data = user_service.get_user_profile(
                        request.user_id)
                    if profile_data:
                        cache.set(cache_key, profile_data,
                                  ttl=get_config().CACHE_TTL_USER_PROFILE)
                if profile_data:
                    user_profile_text = user_service.format_profile_for_prompt(
                        profile_data)
                    comm_pref = (
                        profile_data.get("persona", {})
                        .get("communication_preference", "empathetic")
                    )
                    logger.info(f"👤 Loaded profile for user {request.user_id}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load user profile: {e}")

        # ── Phase 12: Cultural Context Extraction ──
        cultural_context_text = ""
        country_code = "US"
        if request.user_id and profile_data:
            try:
                cultural_svc = get_cultural_service()
                cultural_ctx = cultural_svc.extract_cultural_context(
                    profile_data)
                country_code = cultural_ctx.country_code
                cultural_context_text = cultural_svc.get_cultural_prompt_context(
                    cultural_ctx
                )
            except Exception as e:
                logger.warning(f"Cultural context extraction failed: {e}")

        # ── Phase 16: Health Context from Wearables ──
        health_context_text = ""
        if request.user_id:
            try:
                wearable_svc = get_wearable_service()
                health_context_text = wearable_svc.format_health_context(
                    request.user_id)
                if health_context_text:
                    logger.info(
                        f"💓 Health context loaded for user {request.user_id}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load health context: {e}")

        # Detect emotion and intent from user message
        detection_result = None
        if request.user_id:
            try:
                emotion_svc = get_emotion_service()
                detection_result = emotion_svc.detect(prompt)

                # Auto-store mood entry in user's affective layer
                mood_entry = {
                    "mood": detection_result.emotion,
                    "intensity": detection_result.intensity,
                    "context": prompt[:200],
                    "recorded_at": datetime.now().isoformat(),
                    "detected_via": "text_auto",
                    "confidence": detection_result.confidence,
                    "intent": detection_result.intent,
                }
                user_service = get_user_service()
                user_service.append_mood_entry(request.user_id, mood_entry)
            except Exception as e:
                logger.warning(f"⚠️ Emotion detection failed, continuing: {e}")
                detection_result = None

        # ── Phase 9: Input Safety Screening (fast, no API call) ──
        safety_check = SafetyCheckResult()
        safety_svc = get_safety_service()
        try:
            safety_check = safety_svc.check_input(prompt)
            if safety_check.risk_level != RiskLevel.LOW:
                logger.warning(
                    f"Safety check: risk={safety_check.risk_level.value}, "
                    f"flags={safety_check.flags}"
                )
        except Exception as e:
            logger.warning(f"Safety screening failed, continuing: {e}")

        # If CRITICAL risk, bypass LLM and stream crisis response directly
        if safety_check.risk_level == RiskLevel.CRITICAL:
            crisis_response = safety_svc.get_crisis_intercept_response()

            # Log the safety event
            if request.user_id:
                try:
                    safety_svc.log_safety_event(
                        user_id=request.user_id,
                        session_id=session_id,
                        event_type=SafetyEventType.CRITICAL_INTERCEPT,
                        risk_level=RiskLevel.CRITICAL,
                        message_snippet=prompt[:200],
                        flags=safety_check.flags,
                        action_taken="crisis_intercept",
                    )
                except Exception:
                    pass

                # Phase 20: Auto-escalation for CRITICAL safety (stream)
                try:
                    esc_svc = get_escalation_service()
                    esc_svc.check_and_create_escalation(
                        user_id=request.user_id,
                        safety_risk_level=RiskLevel.CRITICAL.value,
                        safety_event_id=None,
                        dependency_risk_level=DependencyRiskLevel.NONE.value,
                        dependency_score=0,
                    )
                except Exception:
                    pass

            # Update chat history for continuity
            chat_history = session_histories.get(
                session_id, ChatMessageHistory())
            chat_history.add_user_message(prompt)
            chat_history.add_ai_message(crisis_response)
            session_histories[session_id] = chat_history

            async def crisis_event_generator() -> AsyncGenerator[str, None]:
                event_data = json.dumps({
                    "content": crisis_response,
                    "done": False,
                    "session_id": session_id,
                    "response": "",
                    "message_count": 0
                })
                yield f"data: {event_data}\n\n"

                done_data = json.dumps({
                    "content": "",
                    "done": True,
                    "session_id": session_id,
                    "response": crisis_response,
                    "message_count": len(chat_history.messages),
                    "detected_emotion": detection_result.emotion if detection_result else None,
                    "detected_intent": "crisis",
                })
                yield f"data: {done_data}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                crisis_event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        # ── Phase 12: Dependency Detection (fast, no extra API call) ──
        dependency_override_text = ""
        if request.user_id:
            try:
                dep_svc = get_dependency_service()
                dep_metrics = dep_svc.record_interaction(
                    request.user_id, prompt)
                dep_assessment = dep_svc.assess_dependency(
                    request.user_id, dep_metrics, prompt
                )
                if dep_assessment.risk_level != DependencyRiskLevel.NONE:
                    dependency_override_text = dep_svc.format_dependency_for_prompt(
                        dep_assessment
                    )
                    logger.info(
                        f"Dependency: risk={dep_assessment.risk_level.value}, "
                        f"score={dep_assessment.score}, signals={dep_assessment.signals}"
                    )
                elif dep_svc.should_show_self_care_reminder(dep_metrics):
                    dependency_override_text = (
                        "SELF-CARE REMINDER: Subtly include a brief self-care check-in "
                        "in your response (e.g., 'Have you taken a break today?'). "
                        "Keep it natural and brief."
                    )
            except Exception as e:
                logger.warning(f"Dependency check failed, continuing: {e}")

        # Retrieve relevant memories for context
        memory_context_text = ""
        if request.user_id:
            try:
                memory_svc = get_memory_service()
                relevant_memories = memory_svc.search_memories(
                    request.user_id, prompt, limit=5
                )
                memory_context_text = memory_svc.format_memories_for_prompt(
                    relevant_memories
                )
            except Exception as e:
                logger.warning(f"⚠️ Memory retrieval failed, continuing: {e}")

        # Load existing session from Firebase if it exists
        try:
            existing_session = firebase_service.get_chat_session(session_id)
            if existing_session:
                summary = existing_session.get('summary', '')
                session_summaries[session_id] = summary
                logger.info(
                    f"📱 Loaded existing session {session_id} from Firebase")
            else:
                summary = session_summaries.get(session_id, "")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load session from Firebase: {e}")
            summary = session_summaries.get(session_id, "")
            existing_session = None

        # 🔍 RAG: Retrieve relevant context from vector store
        retrieved_context = retrieve_relevant_context(
            prompt, k=5, similarity_threshold=1.1)

        # Build emotion context text for prompt engine
        emotion_context_text = ""
        if detection_result:
            emotion_context_text = get_emotion_service().format_detection_for_prompt(
                detection_result
            )

        # Check for active therapy exercise and due tasks
        active_exercise_text = ""
        due_tasks_text = ""
        if request.user_id:
            try:
                therapy_svc = get_therapy_service()
                active_ex = therapy_svc.get_active_exercise(request.user_id)
                if active_ex:
                    active_exercise_text = therapy_svc.format_exercise_for_prompt(
                        active_ex)
            except Exception as e:
                logger.warning(f"Failed to load active exercise: {e}")

            # Load due/overdue practice tasks for prompt context
            try:
                therapy_svc = get_therapy_service()
                due_tasks = therapy_svc.get_due_tasks(request.user_id)
                if due_tasks:
                    due_tasks_text = therapy_svc.format_due_tasks_for_prompt(
                        due_tasks)
            except Exception as e:
                logger.warning(f"Failed to load due tasks: {e}")

        # Load feedback effectiveness insights for prompt
        feedback_insights_text = ""
        if request.user_id:
            try:
                feedback_svc = get_feedback_service()
                effectiveness = feedback_svc.compute_effectiveness(
                    request.user_id
                )
                if effectiveness.total_outcomes >= 2:
                    feedback_insights_text = (
                        feedback_svc.format_insights_for_prompt(effectiveness)
                    )
            except Exception as e:
                logger.warning(f"Failed to load feedback insights: {e}")

        # Load soundscape suggestion for prompt (when emotion intensity >= 5)
        soundscape_suggestion_text = ""
        if detection_result and detection_result.intensity >= 5:
            try:
                ss_svc = get_soundscape_service()
                soundscape_suggestion_text = ss_svc.format_suggestion_for_prompt(
                    detection_result.emotion, detection_result.intensity
                )
            except Exception as e:
                logger.warning(f"Failed to load soundscape suggestion: {e}")

        # Build safety override text for prompt engine
        safety_override_text = ""
        if safety_check.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) \
                or safety_check.jailbreak_detected:
            safety_override_text = safety_svc.get_safety_prompt_override(
                safety_check
            )

        # Assemble system prompt via prompt engine
        prompt_ctx = PromptContext(
            user_profile_text=user_profile_text,
            memory_context_text=memory_context_text,
            conversation_summary=summary,
            retrieved_rag_context=retrieved_context,
            detection_result=detection_result,
            emotion_context_text=emotion_context_text,
            communication_preference=comm_pref,
            active_exercise_text=active_exercise_text,
            due_tasks_text=due_tasks_text,
            feedback_insights_text=feedback_insights_text,
            soundscape_suggestion_text=soundscape_suggestion_text,
            dependency_override_text=dependency_override_text,
            cultural_context_text=cultural_context_text,
            health_context_text=health_context_text,
            safety_override_text=safety_override_text,
        )
        system_prompt = get_prompt_engine().build_system_prompt(prompt_ctx)

        # Get chat history for context
        chat_history = session_histories.get(session_id, ChatMessageHistory())
        history_messages = []
        for msg in chat_history.messages[-6:]:  # Last 6 messages for context
            if isinstance(msg, HumanMessage):
                history_messages.append(
                    {"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history_messages.append(
                    {"role": "assistant", "content": msg.content})

        # Create messages for OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # Phase 18: Resolve model for A/B routing (stream)
        stream_model_to_use = OPENAI_MODEL
        stream_is_fine_tuned = False
        stream_ab_group = "base"
        try:
            ft_svc = get_finetuning_service()
            stream_model_to_use, stream_is_fine_tuned, stream_ab_group = (
                ft_svc.resolve_model_for_user(request.user_id)
            )
        except Exception:
            stream_model_to_use = OPENAI_MODEL

        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate SSE events with streaming tokens from OpenAI"""
            nonlocal stream_model_to_use, stream_is_fine_tuned, stream_ab_group
            full_response = ""

            try:
                # Phase 1: Resolve tool calls (non-streaming)
                agent_svc = get_chat_agent_service()
                tool_defs = agent_svc.get_tool_definitions()

                try:
                    tool_response = await async_openai_client.chat.completions.create(
                        model=stream_model_to_use,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7,
                        tools=tool_defs,
                        tool_choice="auto",
                    )
                except Exception as model_err:
                    if stream_is_fine_tuned:
                        logger.warning(
                            f"Stream: Fine-tuned model failed, falling back: {model_err}")
                        stream_model_to_use = OPENAI_MODEL
                        stream_is_fine_tuned = False
                        stream_ab_group = "base"
                        tool_response = await async_openai_client.chat.completions.create(
                            model=stream_model_to_use,
                            messages=messages,
                            max_tokens=1000,
                            temperature=0.7,
                            tools=tool_defs,
                            tool_choice="auto",
                        )
                    else:
                        raise

                for _iteration in range(5):
                    tool_msg = tool_response.choices[0].message
                    if not tool_msg.tool_calls:
                        break
                    messages.append(tool_msg)
                    for tool_call in tool_msg.tool_calls:
                        fn_name = tool_call.function.name
                        try:
                            fn_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            fn_args = {}
                        logger.info(f"Stream tool call: {fn_name}({fn_args})")
                        tool_result = agent_svc.execute_tool_call(
                            tool_name=fn_name,
                            tool_args=fn_args,
                            user_id=request.user_id,
                            retrieve_context_fn=retrieve_relevant_context,
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        })
                    tool_response = await async_openai_client.chat.completions.create(
                        model=stream_model_to_use,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7,
                        tools=tool_defs,
                        tool_choice="auto",
                    )

                # Phase 2: Stream the final text response
                final_msg = tool_response.choices[0].message
                final_content = final_msg.content or ""

                if final_content:
                    # Tools resolved (or none needed) — emit as streaming tokens
                    full_response = final_content
                    chunk_size = 4
                    for i in range(0, len(full_response), chunk_size):
                        token = full_response[i:i + chunk_size]
                        event_data = json.dumps({
                            "content": token,
                            "done": False,
                            "session_id": session_id,
                            "response": "",
                            "message_count": 0
                        })
                        yield f"data: {event_data}\n\n"
                else:
                    # No content yet (rare edge case) — stream a fresh call without tools
                    stream = await async_openai_client.chat.completions.create(
                        model=stream_model_to_use,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7,
                        stream=True,
                    )
                    async for chunk in stream:
                        if chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            full_response += token
                            event_data = json.dumps({
                                "content": token,
                                "done": False,
                                "session_id": session_id,
                                "response": "",
                                "message_count": 0
                            })
                            yield f"data: {event_data}\n\n"

                # ── Phase 9: Output Safety Validation (streaming) ──
                try:
                    full_response, output_modified = safety_svc.validate_output(
                        full_response, safety_check, country_code=country_code
                    )
                    if output_modified:
                        logger.info(
                            "Safety: streaming response modified by output validation")
                except Exception as e:
                    logger.warning(f"Output validation failed in stream: {e}")

                # Log safety events for non-low risk
                safety_event_id = None
                if request.user_id and safety_check.risk_level != RiskLevel.LOW:
                    try:
                        event_type = SafetyEventType.CRISIS_DETECTED if safety_check.crisis_detected \
                            else SafetyEventType.HIGH_RISK_INPUT
                        if safety_check.jailbreak_detected:
                            event_type = SafetyEventType.JAILBREAK_ATTEMPT
                        if output_modified:
                            event_type = SafetyEventType.OUTPUT_MODIFIED
                        safety_event_id = safety_svc.log_safety_event(
                            user_id=request.user_id,
                            session_id=session_id,
                            event_type=event_type,
                            risk_level=safety_check.risk_level,
                            message_snippet=prompt[:200],
                            flags=safety_check.flags,
                            action_taken=safety_check.action_taken,
                        )
                    except Exception:
                        pass

                    # Phase 20: Auto-escalation check for non-LOW safety events (stream)
                    try:
                        _dep_risk = DependencyRiskLevel.NONE.value
                        _dep_score = 0
                        if 'dep_assessment' in dir():
                            _dep_risk = dep_assessment.risk_level.value
                            _dep_score = dep_assessment.score
                        esc_svc = get_escalation_service()
                        esc_svc.check_and_create_escalation(
                            user_id=request.user_id,
                            safety_risk_level=safety_check.risk_level.value,
                            safety_event_id=safety_event_id,
                            dependency_risk_level=_dep_risk,
                            dependency_score=_dep_score,
                        )
                    except Exception:
                        pass

                # Update chat history after streaming completes
                chat_history.add_user_message(prompt)
                chat_history.add_ai_message(full_response)
                session_histories[session_id] = chat_history

                # Store/Update session in Firebase (async-safe)
                try:
                    current_messages = chat_history.messages
                    current_summary = session_summaries.get(session_id, "")

                    if existing_session:
                        firebase_service.update_chat_session(
                            session_id, current_messages, current_summary)
                    else:
                        firebase_service.store_chat_session(
                            session_id, current_messages, current_summary)
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to store session in Firebase: {e}")

                # Link session to user if user_id was provided
                if request.user_id:
                    try:
                        firebase_service.link_session_to_user(
                            session_id, request.user_id)
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to link session to user: {e}")

                # Extract and store memories from this conversation turn
                if request.user_id:
                    try:
                        memory_svc = get_memory_service()
                        memory_svc.extract_memories(
                            request.user_id, prompt, full_response)
                    except Exception as e:
                        logger.warning(f"⚠️ Memory extraction failed: {e}")

                # Summarize if over token limit
                full_text = f"{prompt} {full_response}"
                if count_tokens(full_text) > 8000:
                    logger.info("🔁 Summarizing chat history...")
                    try:
                        summary_prompt = f"Please summarize the following conversation history in a concise way, focusing on key topics and decisions:\n\n{full_text}"
                        refined_summary = summary_llm.invoke(
                            summary_prompt).content
                        session_summaries[session_id] = refined_summary
                        firebase_service.update_session_summary(
                            session_id, refined_summary)
                    except Exception as e:
                        logger.warning(f"⚠️ Summarization failed: {e}")

                # Send completion event - SAME FORMAT as streaming messages
                done_data = json.dumps({
                    "content": "",
                    "done": True,
                    "session_id": session_id,
                    "response": full_response,
                    "message_count": len(chat_history.messages),
                    "detected_emotion": detection_result.emotion if detection_result else None,
                    "detected_intent": detection_result.intent if detection_result else None,
                })
                yield f"data: {done_data}\n\n"
                yield "data: [DONE]\n\n"

                logger.info(f"✅ Streaming completed for session {session_id}")

            except APIConnectionError as e:
                logger.error(f"❌ OpenAI network error: {e}")
                error_data = json.dumps(
                    {"type": "error", "message": f"OpenAI network error: {e}"})
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
            except RateLimitError as e:
                logger.error(f"❌ OpenAI rate limit: {e}")
                error_data = json.dumps(
                    {"type": "error", "message": "OpenAI rate limit hit"})
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"❌ Streaming error: {e}")
                error_data = json.dumps({"type": "error", "message": str(e)})
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except Exception as e:
        logger.error(f"❌ Error in streaming chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint


@app.get("/health")
async def health_check():
    """Health check endpoint with basic status."""
    try:
        firebase_service = get_firebase_service()
        sessions = firebase_service.list_sessions(limit=1)
        metrics = get_metrics_collector()
        summary = metrics.get_summary()
        return {
            "status": "healthy",
            "firebase": "connected",
            "vector_store": "loaded",
            "model": OPENAI_MODEL,
            "environment": get_config().ENVIRONMENT,
            "total_requests": sum(
                ep.get("count", 0)
                for ep in summary.get("endpoints", {}).values()
            ),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "firebase": "disconnected",
            "error": str(e)
        }


@app.get("/health/detailed")
async def health_detailed():
    """Detailed health check with cache stats, FAISS info, and memory usage."""
    cache = get_cache()
    result = {
        "status": "healthy",
        "environment": get_config().ENVIRONMENT,
        "model": OPENAI_MODEL,
        "services": {},
        "cache": cache.stats(),
        "faiss": {
            "documents": len(DOCS),
            "dimension": VectorStore.index.d if VectorStore else None,
        },
    }
    # Check Firebase
    try:
        get_firebase_service().list_sessions(limit=1)
        result["services"]["firebase"] = "ok"
    except Exception as e:
        result["services"]["firebase"] = f"error: {e}"
        result["status"] = "degraded"
    # Check OpenAI
    try:
        client = OpenAI()
        client.models.list()
        result["services"]["openai"] = "ok"
    except Exception as e:
        result["services"]["openai"] = f"error: {e}"
        result["status"] = "degraded"
    # Check FAISS
    try:
        result["services"]["faiss"] = "ok" if VectorStore else "not loaded"
    except Exception:
        result["services"]["faiss"] = "error"
        result["status"] = "degraded"
    # Memory stats (optional psutil)
    try:
        import psutil
        proc = psutil.Process()
        result["memory"] = {
            "rss_mb": round(proc.memory_info().rss / 1024 / 1024, 1),
            "percent": round(proc.memory_percent(), 1),
        }
    except ImportError:
        result["memory"] = "psutil not installed"
    return result


@app.get("/metrics")
async def get_metrics():
    """Return per-endpoint request metrics with latency percentiles."""
    return get_metrics_collector().get_summary()


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint that generates a new session ID"""
    session_id = str(uuid.uuid4())
    return JSONResponse(content={
        "session_id": session_id,
        "status": "success",
        "message": "New session created",
        "timestamp": datetime.now().isoformat()
    })

# Chat interface


@app.get("/chat-interface", response_class=HTMLResponse)
async def chat_interface():
    """Simple HTML chat interface"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lucille - Self-Care Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 20px; margin: 20px 0; }
            .input-container { display: flex; gap: 10px; }
            input[type="text"] { flex: 1; padding: 10px; font-size: 16px; }
            button { padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>Lucille - Self-Care Chatbot</h1>
        <p>Welcome! I'm here to help with self-care and wellbeing advice.</p>
        <div class="chat-container" id="chatContainer"></div>
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Type your message here..." />
            <button onclick="sendMessage()">Send</button>
        </div>
        
        <script>
            let sessionId = Math.random().toString(36).substring(7);
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message) return;
                
                const chatContainer = document.getElementById('chatContainer');
                chatContainer.innerHTML += `<div><strong>You:</strong> ${message}</div>`;
                input.value = '';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: message, session_id: sessionId })
                    });
                    
                    const data = await response.json();
                    chatContainer.innerHTML += `<div><strong>Lucille:</strong> ${data.response}</div>`;
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                } catch (error) {
                    chatContainer.innerHTML += `<div style="color: red;"><strong>Error:</strong> ${error.message}</div>`;
                }
            }
            
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendMessage();
            });
        </script>
    </body>
    </html>
    """)

# Session management endpoints


@app.get("/chat/{session_id}", response_model=ChatResponse)
async def get_chat_history(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    """Retrieve chat history for a session (owner or admin only)"""
    try:
        # Validate session ID
        session_id = validate_session_id(session_id)
        firebase_service = get_firebase_service()
        session_data = firebase_service.get_chat_session(session_id)

        if not session_data:
            raise HTTPException(
                status_code=404, detail="No chat history found.")

        # Ownership check. Sessions are stamped with user_id by
        # link_session_to_user() on the /chat path. Sessions predating that
        # field have no owner — treat those as admin-only rather than
        # world-readable, since these are therapy transcripts.
        owner = session_data.get('user_id')
        if not _is_admin(user) and owner != user.get('uid'):
            raise HTTPException(
                status_code=403,
                detail="You can only access your own sessions.")

        messages = session_data.get('messages', [])
        conversation_history = [msg.get('content', '') for msg in messages]

        return ChatResponse(
            session_id=session_id,
            response="Chat history retrieved successfully",
            conversation=conversation_history,
            status="success",
            timestamp=datetime.now().isoformat(),
            message_count=len(conversation_history)
        )
    except HTTPException:
        # Deliberate status codes (401/403/404) must not be reclassified as 500
        # by the generic handler below.
        raise
    except Exception as e:
        print(f"❌ Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/{session_id}")
async def delete_chat_session(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a chat session (owner or admin only)"""
    try:
        # Validate session ID
        session_id = validate_session_id(session_id)
        firebase_service = get_firebase_service()

        # Resolve ownership BEFORE deleting — see note in get_chat_history.
        session_data = firebase_service.get_chat_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        owner = session_data.get('user_id')
        if not _is_admin(user) and owner != user.get('uid'):
            raise HTTPException(
                status_code=403,
                detail="You can only delete your own sessions.")

        success = firebase_service.delete_chat_session(session_id)

        if success:
            # Clear from memory
            if session_id in session_histories:
                del session_histories[session_id]
            if session_id in session_summaries:
                del session_summaries[session_id]

            return JSONResponse(content={
                "session_id": session_id,
                "status": "success",
                "message": f"Session {session_id} deleted successfully",
                "timestamp": datetime.now().isoformat()
            })
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/")
async def list_sessions(
    limit: int = 100,
    user: dict = Depends(get_current_user),
):
    """
    List chat sessions.

    Scoped by identity: admins see every session, ordinary users see only
    their own. A bare Depends(get_current_user) would not be enough here —
    list_sessions() reads the whole chat_sessions collection, so any logged-in
    account could enumerate every user's therapy transcripts.
    """
    try:
        firebase_service = get_firebase_service()
        if _is_admin(user):
            sessions = firebase_service.list_sessions(limit)
        else:
            sessions = firebase_service.get_user_sessions(
                user.get('uid', ''), limit)
        return JSONResponse(content={
            "sessions": sessions,
            "count": len(sessions),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Session validation endpoint


@app.get("/session/{session_id}/validate")
async def validate_session(session_id: str):
    """Validate if a session exists"""
    try:
        # Validate session ID format
        session_id = validate_session_id(session_id)
        firebase_service = get_firebase_service()
        session_data = firebase_service.get_chat_session(session_id)

        return JSONResponse(content={
            "session_id": session_id,
            "valid": session_data is not None,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "session_id": session_id,
                "valid": False,
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# ── User Profile Endpoints ──────────────────────────────


@app.post("/users/onboard", response_model=OnboardingResponse)
async def onboard_user(request: OnboardingRequest):
    """
    Onboarding endpoint. Creates a user profile from initial data.
    If user_id is not provided, generates one.
    """
    try:
        user_service = get_user_service()

        user_id = request.user_id or str(uuid.uuid4())

        # Check if profile already exists
        existing = user_service.get_user_profile(user_id)
        if existing:
            return JSONResponse(
                status_code=409,
                content={
                    "user_id": user_id,
                    "status": "error",
                    "message": "User profile already exists. Use PUT /users/{user_id} to update.",
                    "timestamp": datetime.now().isoformat()
                }
            )

        # Build goals from simple title strings
        goals = [
            Goal(goal_id=str(uuid.uuid4()), title=g, status="active")
            for g in request.goals
        ] if request.goals else []

        profile = UserProfile(
            user_id=user_id,
            persona=PersonaLayer(
                display_name=request.display_name,
                personality_traits=request.personality_traits,
                communication_preference=request.communication_preference,
                age_range=request.age_range,
                interests=request.interests,
            ),
            affective=AffectiveLayer(
                current_mood=request.current_mood,
            ),
            motivational=MotivationalLayer(
                core_values=request.core_values,
                goals=goals,
            ),
            behavioral=BehavioralLayer(
                sleep_pattern=request.sleep_pattern,
                exercise_frequency=request.exercise_frequency,
            ),
            onboarding_completed=True,
        )

        created_id = user_service.create_user_profile(profile.model_dump())
        if not created_id:
            raise HTTPException(
                status_code=500, detail="Failed to create user profile")

        # Invalidate any cached profile for this user
        get_cache().invalidate(user_profile_key(created_id))

        return OnboardingResponse(
            user_id=created_id,
            status="success",
            message="User profile created successfully",
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in onboarding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}", response_model=UserProfileResponse)
async def get_user_profile(user_id: str):
    """Get a user's complete profile"""
    try:
        user_service = get_user_service()
        profile_data = user_service.get_user_profile(user_id)

        if not profile_data:
            raise HTTPException(
                status_code=404, detail="User profile not found")

        profile = UserProfile(**profile_data)
        return UserProfileResponse(
            user_id=user_id,
            profile=profile,
            status="success",
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/users/{user_id}", response_model=UserProfileResponse)
async def update_user_profile(user_id: str, request: UserProfileUpdateRequest):
    """Update specific layers of a user profile (partial update)"""
    try:
        user_service = get_user_service()

        existing = user_service.get_user_profile(user_id)
        if not existing:
            raise HTTPException(
                status_code=404, detail="User profile not found")

        update_data = {}
        if request.cognitive is not None:
            update_data["cognitive"] = request.cognitive.model_dump()
        if request.affective is not None:
            update_data["affective"] = request.affective.model_dump()
        if request.behavioral is not None:
            update_data["behavioral"] = request.behavioral.model_dump()
        if request.motivational is not None:
            update_data["motivational"] = request.motivational.model_dump()
        if request.persona is not None:
            update_data["persona"] = request.persona.model_dump()

        if not update_data:
            raise HTTPException(
                status_code=400, detail="No update data provided")

        success = user_service.update_user_profile(user_id, update_data)
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to update profile")

        # Invalidate cached profile
        get_cache().invalidate(user_profile_key(user_id))

        # Return the updated profile
        updated = user_service.get_user_profile(user_id)
        profile = UserProfile(**updated)
        return UserProfileResponse(
            user_id=user_id,
            profile=profile,
            status="success",
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/users/{user_id}")
async def delete_user_profile(user_id: str):
    """Delete a user profile"""
    try:
        user_service = get_user_service()
        success = user_service.delete_user_profile(user_id)
        if not success:
            raise HTTPException(
                status_code=404, detail="User profile not found or could not be deleted")
        return JSONResponse(content={
            "user_id": user_id,
            "status": "success",
            "message": "User profile deleted",
            "timestamp": datetime.now().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users/{user_id}/mood")
async def record_mood(user_id: str, mood_entry: MoodEntry):
    """Append a mood entry to a user's affective layer"""
    try:
        user_service = get_user_service()
        success = user_service.append_mood_entry(
            user_id, mood_entry.model_dump())
        if not success:
            raise HTTPException(
                status_code=404, detail="User not found or update failed")
        return JSONResponse(content={
            "user_id": user_id,
            "status": "success",
            "message": "Mood recorded",
            "timestamp": datetime.now().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error recording mood: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users/{user_id}/mood/analyze-image",
          dependencies=[Depends(require_same_user())])
async def analyze_mood_image(
    user_id: str,
    file: UploadFile = File(...),
    store: bool = True,
):
    """Analyze a user's facial emotion from an uploaded image (OpenAI Vision).

    Replaces the legacy ViT classifier (vit_emotion_api/). Detects the dominant
    emotion and, by default, records it as a mood entry with
    detected_via='image_auto'. Pass ?store=false to analyze without saving.
    """
    # Validate the upload is an image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file")

    max_bytes = get_config().MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {get_config().MAX_IMAGE_SIZE_MB}MB limit")

    # Classify emotion via OpenAI vision
    try:
        emotion_svc = get_emotion_service()
        detection = emotion_svc.detect_from_image(contents, file.content_type)
    except Exception as e:
        logger.error(f"❌ Image mood analysis failed: {e}")
        raise HTTPException(
            status_code=502, detail="Image mood analysis failed")

    # Optionally store as a mood entry (detected_via='image_auto')
    stored = False
    if store:
        try:
            user_service = get_user_service()
            mood_entry = {
                "mood": detection.emotion,
                "intensity": detection.intensity,
                "context": "Detected from uploaded image",
                "recorded_at": datetime.now().isoformat(),
                "detected_via": "image_auto",
                "confidence": detection.confidence,
                "intent": None,
            }
            stored = user_service.append_mood_entry(user_id, mood_entry)
        except Exception as e:
            logger.warning(f"⚠️ Failed to store image mood entry: {e}")

    return JSONResponse(content={
        "user_id": user_id,
        "status": "success",
        "detected_emotion": detection.emotion,
        "intensity": detection.intensity,
        "confidence": detection.confidence,
        "detected_via": "image_auto",
        "stored": stored,
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/users/{user_id}/sessions")
async def get_user_sessions(user_id: str, limit: int = 50):
    """List all chat sessions belonging to a user"""
    try:
        firebase_service = get_firebase_service()
        sessions = firebase_service.get_user_sessions(user_id, limit)
        return JSONResponse(content={
            "user_id": user_id,
            "sessions": sessions,
            "count": len(sessions),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Error listing user sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Memory Endpoints ──────────────────────────────────────

@app.get("/users/{user_id}/memories")
async def list_memories(
    user_id: str,
    memory_type: Optional[str] = None,
    limit: int = 50,
):
    """List memories for a user, optionally filtered by type"""
    try:
        memory_svc = get_memory_service()
        memories = memory_svc.get_memories(
            user_id, memory_type=memory_type, limit=limit)
        return JSONResponse(content={
            "user_id": user_id,
            "memories": memories,
            "count": len(memories),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error listing memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users/{user_id}/memories")
async def create_memory(user_id: str, request: MemoryCreateRequest):
    """Manually add a memory for a user"""
    try:
        memory_svc = get_memory_service()
        memory_data = {
            "content": request.content,
            "memory_type": request.memory_type.value,
            "importance": request.importance,
            "tags": request.tags,
            "source": "manual",
        }
        memory_id = memory_svc.store_memory(user_id, memory_data)
        if not memory_id:
            raise HTTPException(
                status_code=500, detail="Failed to store memory")

        return JSONResponse(content={
            "user_id": user_id,
            "memory_id": memory_id,
            "status": "success",
            "message": "Memory stored",
            "timestamp": datetime.now().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users/{user_id}/memories/search")
async def search_memories(user_id: str, request: MemorySearchRequest):
    """Semantic search over a user's memories"""
    try:
        memory_svc = get_memory_service()
        results = memory_svc.search_memories(
            user_id, request.query, limit=request.limit)
        return JSONResponse(content={
            "user_id": user_id,
            "results": results,
            "count": len(results),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/users/{user_id}/memories/{memory_id}")
async def delete_memory(user_id: str, memory_id: str):
    """Delete a specific memory"""
    try:
        memory_svc = get_memory_service()
        success = memory_svc.delete_memory(user_id, memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return JSONResponse(content={
            "user_id": user_id,
            "memory_id": memory_id,
            "status": "success",
            "message": "Memory deleted",
            "timestamp": datetime.now().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users/{user_id}/memories/consolidate")
async def consolidate_memories(user_id: str):
    """Remove old, low-importance, unaccessed memories"""
    try:
        memory_svc = get_memory_service()
        removed = memory_svc.consolidate_memories(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "removed_count": removed,
            "status": "success",
            "message": f"Consolidated {removed} old memories",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error consolidating memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Therapy Exercise Endpoints ────────────────────────────

@app.get("/therapy/exercises")
async def list_exercises(modality: Optional[str] = None):
    """List available therapy exercises, optionally filtered by modality"""
    try:
        therapy_svc = get_therapy_service()
        exercises = therapy_svc.list_exercises(modality=modality)
        return JSONResponse(content={
            "exercises": exercises,
            "count": len(exercises),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error listing exercises: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/therapy/exercises/{exercise_id}")
async def get_exercise(exercise_id: str):
    """Get details of a specific exercise"""
    try:
        therapy_svc = get_therapy_service()
        exercise = therapy_svc.get_exercise(exercise_id)
        if exercise is None:
            raise HTTPException(status_code=404, detail="Exercise not found")
        return JSONResponse(content={
            "exercise": exercise.model_dump(),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting exercise: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/therapy/recommend/{user_id}")
async def recommend_exercises(user_id: str, limit: int = 3):
    """Recommend exercises based on user's current emotional state"""
    try:
        # Get latest emotion detection for the user
        emotion = "neutral"
        intent = "casual_chat"
        user_service = get_user_service()
        profile_data = user_service.get_user_profile(user_id)
        if profile_data:
            affective = profile_data.get("affective", {})
            mood_history = affective.get("mood_history", [])
            if mood_history:
                latest = mood_history[-1]
                emotion = latest.get("mood", "neutral")
                intent = latest.get("intent", "casual_chat") or "casual_chat"

        # Phase 7: Load effectiveness scores for adaptive recommendations
        effectiveness_scores = None
        try:
            feedback_svc = get_feedback_service()
            effectiveness = feedback_svc.compute_effectiveness(user_id)
            if effectiveness.total_outcomes >= 2:
                effectiveness_scores = {
                    "modality_scores": effectiveness.modality_scores,
                    "exercise_scores": effectiveness.exercise_scores,
                }
        except Exception as e:
            logger.warning(
                f"Failed to load effectiveness for recommendations: {e}"
            )

        therapy_svc = get_therapy_service()
        recs = therapy_svc.recommend_exercises(
            emotion=emotion,
            intent=intent,
            limit=limit,
            effectiveness_scores=effectiveness_scores,
        )
        return JSONResponse(content={
            "user_id": user_id,
            "detected_emotion": emotion,
            "detected_intent": intent,
            "recommendations": [r.model_dump() for r in recs],
            "count": len(recs),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error recommending exercises: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/therapy/{user_id}/start", response_model=StartExerciseResponse)
async def start_exercise(user_id: str, request: StartExerciseRequest):
    """Start a therapy exercise for a user"""
    try:
        therapy_svc = get_therapy_service()

        # Check for already active exercise
        active = therapy_svc.get_active_exercise(user_id)
        if active:
            raise HTTPException(
                status_code=409,
                detail=f"User already has an active exercise: {active.title}. "
                f"Complete or abandon it first."
            )

        session = therapy_svc.start_exercise(user_id, request.exercise_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Exercise not found")

        exercise = therapy_svc.get_exercise(request.exercise_id)
        first_step = exercise.steps[0] if exercise and exercise.steps else ""

        # Phase 8: Look up suggested soundscape for this exercise
        suggested_soundscape = None
        if request.exercise_id in EXERCISE_SOUNDSCAPE_MAP:
            ss_id = EXERCISE_SOUNDSCAPE_MAP[request.exercise_id]
            ss_template = get_soundscape_service().get_soundscape(ss_id)
            if ss_template:
                suggested_soundscape = {
                    "soundscape_id": ss_template.soundscape_id,
                    "title": ss_template.title,
                    "category": ss_template.category.value,
                    "audio_url": ss_template.audio_url,
                    "icon": ss_template.icon,
                }

        return StartExerciseResponse(
            session_id=session.session_id,
            exercise_id=session.exercise_id,
            modality=session.modality,
            title=session.title,
            total_steps=session.total_steps,
            first_step=first_step,
            status="success",
            timestamp=datetime.now().isoformat(),
            suggested_soundscape=suggested_soundscape,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting exercise: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/therapy/{user_id}/advance/{session_id}")
async def advance_exercise(user_id: str, session_id: str, note: str = ""):
    """Advance to the next step of an active exercise"""
    try:
        therapy_svc = get_therapy_service()
        session = therapy_svc.advance_exercise(
            user_id, session_id, user_note=note)
        if session is None:
            raise HTTPException(
                status_code=404, detail="Exercise session not found")

        # Get the next step text
        next_step = ""
        if session.status == "active":
            exercise = therapy_svc.get_exercise(session.exercise_id)
            if exercise and session.current_step < len(exercise.steps):
                next_step = exercise.steps[session.current_step]

        return JSONResponse(content={
            "session_id": session.session_id,
            "current_step": session.current_step,
            "total_steps": session.total_steps,
            "status": session.status,
            "next_step": next_step,
            "timestamp": datetime.now().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error advancing exercise: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/therapy/{user_id}/abandon/{session_id}")
async def abandon_exercise(user_id: str, session_id: str):
    """Abandon an active exercise"""
    try:
        therapy_svc = get_therapy_service()
        success = therapy_svc.abandon_exercise(user_id, session_id)
        if not success:
            raise HTTPException(
                status_code=404, detail="Exercise session not found")
        return JSONResponse(content={
            "session_id": session_id,
            "status": "abandoned",
            "message": "Exercise abandoned",
            "timestamp": datetime.now().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error abandoning exercise: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/therapy/{user_id}/active")
async def get_active_exercise(user_id: str):
    """Get the user's currently active exercise, if any"""
    try:
        therapy_svc = get_therapy_service()
        active = therapy_svc.get_active_exercise(user_id)
        if active is None:
            return JSONResponse(content={
                "user_id": user_id,
                "active_exercise": None,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            })

        # Get current step text
        current_step_text = ""
        exercise = therapy_svc.get_exercise(active.exercise_id)
        if exercise and active.current_step < len(exercise.steps):
            current_step_text = exercise.steps[active.current_step]

        return JSONResponse(content={
            "user_id": user_id,
            "active_exercise": {
                "session_id": active.session_id,
                "exercise_id": active.exercise_id,
                "modality": active.modality,
                "title": active.title,
                "current_step": active.current_step,
                "total_steps": active.total_steps,
                "current_step_text": current_step_text,
                "started_at": active.started_at,
            },
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting active exercise: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/therapy/{user_id}/history")
async def get_exercise_history(user_id: str, limit: int = 20):
    """Get a user's exercise history"""
    try:
        therapy_svc = get_therapy_service()
        history = therapy_svc.get_exercise_history(user_id, limit=limit)
        return JSONResponse(content={
            "user_id": user_id,
            "history": history,
            "count": len(history),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting exercise history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Task / Homework Endpoints ────────────────────────────

@app.get("/therapy/{user_id}/tasks")
async def list_tasks(user_id: str, status: Optional[str] = None, limit: int = 20):
    """List practice tasks for a user, optionally filtered by status"""
    try:
        therapy_svc = get_therapy_service()
        tasks = therapy_svc.get_tasks(
            user_id, status_filter=status, limit=limit)
        return JSONResponse(content={
            "user_id": user_id,
            "tasks": tasks,
            "count": len(tasks),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/therapy/{user_id}/tasks/due")
async def get_due_tasks_endpoint(user_id: str):
    """Get pending/in_progress tasks that are due or overdue"""
    try:
        therapy_svc = get_therapy_service()
        tasks = therapy_svc.get_due_tasks(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "due_tasks": tasks,
            "count": len(tasks),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting due tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/therapy/{user_id}/tasks")
async def create_task(user_id: str, request: CreateTaskRequest):
    """Manually create a practice task for a user"""
    try:
        therapy_svc = get_therapy_service()

        # Validate exercise exists
        exercise = therapy_svc.get_exercise(request.source_exercise_id)
        if exercise is None:
            raise HTTPException(
                status_code=404, detail="Source exercise not found")

        task = PracticeTask(
            user_id=user_id,
            source_exercise_id=request.source_exercise_id,
            modality=exercise.modality.value,
            title=request.title,
            description=request.description,
            due_date=request.due_date,
            target_count=request.target_count,
        )

        task_id = therapy_svc.create_task(user_id, task)
        if not task_id:
            raise HTTPException(
                status_code=500, detail="Failed to create task")

        return JSONResponse(content={
            "user_id": user_id,
            "task_id": task_id,
            "status": "success",
            "message": "Task created",
            "timestamp": datetime.now().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/therapy/{user_id}/tasks/{task_id}")
async def update_task(user_id: str, task_id: str, request: UpdateTaskRequest):
    """Update a task's status, progress count, or add a note"""
    try:
        therapy_svc = get_therapy_service()

        updates = {}
        if request.status is not None:
            updates["status"] = request.status.value
        if request.completed_count is not None:
            updates["completed_count"] = request.completed_count
        if request.note is not None:
            updates["note"] = request.note

        if not updates:
            raise HTTPException(
                status_code=400, detail="No update fields provided")

        result = therapy_svc.update_task(user_id, task_id, updates)
        if result is None:
            raise HTTPException(status_code=404, detail="Task not found")

        return JSONResponse(content={
            "user_id": user_id,
            "task_id": task_id,
            "task": result,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Progress Tracking Endpoints ──────────────────────────

@app.get("/therapy/{user_id}/progress")
async def get_progress(user_id: str):
    """Get computed progress analytics for a user"""
    try:
        progress_svc = get_progress_service()
        summary = progress_svc.get_progress_summary(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "progress": summary.model_dump(),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error computing progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Feedback & Closed-Loop Endpoints ──────────────────────

@app.post("/feedback/{user_id}/response")
async def submit_response_feedback(
    user_id: str, request: SubmitFeedbackRequest
):
    """Submit feedback on a chat response."""
    try:
        # Enrich with latest detected emotion/intent from user profile
        detected_emotion = ""
        detected_intent = ""
        try:
            user_service = get_user_service()
            profile_data = user_service.get_user_profile(user_id)
            if profile_data:
                mood_history = (
                    profile_data.get("affective", {})
                    .get("mood_history", [])
                )
                if mood_history:
                    latest = mood_history[-1]
                    detected_emotion = latest.get("mood", "")
                    detected_intent = latest.get("intent", "")
        except Exception:
            pass

        feedback = ResponseFeedback(
            user_id=user_id,
            session_id=request.session_id,
            message_index=request.message_index,
            rating=request.rating,
            comment=request.comment,
            detected_emotion=detected_emotion,
            detected_intent=detected_intent,
        )

        feedback_svc = get_feedback_service()
        feedback_id = feedback_svc.store_response_feedback(user_id, feedback)
        if feedback_id is None:
            raise HTTPException(
                status_code=500, detail="Failed to store feedback"
            )

        # Phase 18: Link feedback to model performance record
        try:
            ft_svc = get_finetuning_service()
            if ft_svc.is_available:
                ft_svc.update_performance_feedback(
                    session_id=request.session_id,
                    rating=request.rating.value,
                )
        except Exception as e:
            logger.warning(f"FT feedback linking failed: {e}")

        return JSONResponse(content={
            "user_id": user_id,
            "feedback_id": feedback_id,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting response feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback/{user_id}/exercise-outcome")
async def submit_exercise_outcome(
    user_id: str, request: SubmitOutcomeRequest
):
    """Submit outcome data after completing an exercise."""
    try:
        # Look up the exercise session to get exercise_id and modality
        therapy_svc = get_therapy_service()
        exercise_id = ""
        modality = ""

        if therapy_svc.db is not None:
            try:
                doc_ref = (
                    therapy_svc.db.collection(therapy_svc.COLLECTION)
                    .document(user_id)
                    .collection("sessions")
                    .document(request.session_id)
                )
                doc = doc_ref.get()
                if doc.exists:
                    data = doc.to_dict()
                    exercise_id = data.get("exercise_id", "")
                    modality = data.get("modality", "")
            except Exception as e:
                logger.warning(f"Failed to look up exercise session: {e}")

        if not exercise_id:
            raise HTTPException(
                status_code=404,
                detail=f"Exercise session {request.session_id} not found",
            )

        outcome = ExerciseOutcome(
            user_id=user_id,
            session_id=request.session_id,
            exercise_id=exercise_id,
            modality=modality,
            mood_before=request.mood_before,
            mood_after=request.mood_after,
            helpfulness=request.helpfulness,
            would_repeat=request.would_repeat,
            comment=request.comment,
        )

        feedback_svc = get_feedback_service()
        outcome_id = feedback_svc.store_exercise_outcome(user_id, outcome)
        if outcome_id is None:
            raise HTTPException(
                status_code=500, detail="Failed to store exercise outcome"
            )

        return JSONResponse(content={
            "user_id": user_id,
            "outcome_id": outcome_id,
            "exercise_id": exercise_id,
            "modality": modality,
            "mood_delta": request.mood_after - request.mood_before,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting exercise outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/{user_id}/history")
async def get_feedback_history(user_id: str, limit: int = 20):
    """Get both response feedback and exercise outcomes for a user."""
    try:
        feedback_svc = get_feedback_service()
        response_feedback = feedback_svc.get_response_feedback(
            user_id, limit=limit
        )
        exercise_outcomes = feedback_svc.get_exercise_outcomes(
            user_id, limit=limit
        )

        return JSONResponse(content={
            "user_id": user_id,
            "response_feedback": response_feedback,
            "exercise_outcomes": exercise_outcomes,
            "total_response_feedback": len(response_feedback),
            "total_exercise_outcomes": len(exercise_outcomes),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error fetching feedback history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/{user_id}/effectiveness")
async def get_effectiveness(user_id: str):
    """Get computed effectiveness profile for a user."""
    try:
        feedback_svc = get_feedback_service()
        profile = feedback_svc.compute_effectiveness(user_id)

        return JSONResponse(content={
            "user_id": user_id,
            "effectiveness": profile.model_dump(),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error computing effectiveness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Soundscape & Audio Engine Endpoints ──────────────────

@app.get("/soundscapes")
async def list_soundscapes(category: Optional[str] = None):
    """List all soundscapes, optionally filtered by category."""
    try:
        ss_svc = get_soundscape_service()
        soundscapes = ss_svc.list_soundscapes(category=category)
        return JSONResponse(content={
            "soundscapes": [ss.model_dump() for ss in soundscapes],
            "count": len(soundscapes),
            "category_filter": category,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error listing soundscapes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/soundscapes/categories")
async def list_soundscape_categories():
    """List available soundscape categories with counts."""
    try:
        ss_svc = get_soundscape_service()
        all_ss = ss_svc.list_soundscapes()
        category_counts = defaultdict(int)
        for ss in all_ss:
            category_counts[ss.category.value] += 1
        return JSONResponse(content={
            "categories": dict(category_counts),
            "total": len(all_ss),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error listing soundscape categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/soundscapes/recommend/{user_id}")
async def recommend_soundscapes(
    user_id: str, exercise_id: Optional[str] = None
):
    """Recommend soundscapes based on user's latest emotion."""
    try:
        # Get user's latest emotion from profile
        user_svc = get_user_service()
        profile = user_svc.get_user_profile(user_id)
        emotion = "neutral"
        intent = "casual_chat"
        if profile:
            emotion = profile.affective.current_mood
            if profile.affective.mood_history:
                latest = profile.affective.mood_history[-1]
                emotion = latest.mood
                intent = latest.intent or "casual_chat"

        ss_svc = get_soundscape_service()
        recs = ss_svc.recommend_soundscapes(
            emotion=emotion,
            intent=intent,
            exercise_id=exercise_id,
            limit=3,
        )

        return JSONResponse(content={
            "user_id": user_id,
            "detected_emotion": emotion,
            "exercise_id": exercise_id,
            "recommendations": [r.model_dump() for r in recs],
            "count": len(recs),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error recommending soundscapes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/soundscapes/{soundscape_id}")
async def get_soundscape(soundscape_id: str):
    """Get a specific soundscape by ID."""
    try:
        ss_svc = get_soundscape_service()
        ss = ss_svc.get_soundscape(soundscape_id)
        if ss is None:
            raise HTTPException(
                status_code=404, detail=f"Soundscape '{soundscape_id}' not found"
            )
        return JSONResponse(content={
            "soundscape": ss.model_dump(),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting soundscape: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/soundscapes/{user_id}/start")
async def start_soundscape(user_id: str, request: StartSoundscapeRequest):
    """Start a soundscape listening session."""
    try:
        ss_svc = get_soundscape_service()
        session = ss_svc.start_session(user_id, request.soundscape_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Soundscape '{request.soundscape_id}' not found"
            )
        return JSONResponse(content={
            "session": session.model_dump(),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting soundscape: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/soundscapes/{user_id}/stop/{session_id}")
async def stop_soundscape(user_id: str, session_id: str):
    """Stop a soundscape listening session."""
    try:
        ss_svc = get_soundscape_service()
        session = ss_svc.stop_session(user_id, session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Soundscape session '{session_id}' not found"
            )
        return JSONResponse(content={
            "session": session.model_dump(),
            "duration_seconds": session.duration_seconds,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping soundscape: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/soundscapes/{user_id}/history")
async def get_soundscape_history(user_id: str, limit: int = 20):
    """Get soundscape session history for a user."""
    try:
        ss_svc = get_soundscape_service()
        sessions = ss_svc.get_session_history(user_id, limit=limit)
        return JSONResponse(content={
            "user_id": user_id,
            "sessions": sessions,
            "count": len(sessions),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error fetching soundscape history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Safety & Ethics Endpoints ────────────────────────────

@app.get("/safety/resources")
async def list_crisis_resources():
    """List all available crisis resources."""
    try:
        safety_svc = get_safety_service()
        resources = safety_svc.get_crisis_resources()
        return JSONResponse(content={
            "resources": [r.model_dump() for r in resources],
            "count": len(resources),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error listing crisis resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/safety/{user_id}/audit")
async def get_safety_audit(user_id: str, limit: int = 50):
    """Get safety audit log for a user."""
    try:
        safety_svc = get_safety_service()
        events = safety_svc.get_safety_audit(user_id, limit=limit)
        return JSONResponse(content={
            "user_id": user_id,
            "events": events,
            "count": len(events),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error fetching safety audit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/safety/check")
async def manual_safety_check(request: SafetyCheckRequest):
    """Manual safety check on text (for testing/admin use)."""
    try:
        safety_svc = get_safety_service()
        result = safety_svc.check_input(request.text)
        return JSONResponse(content={
            "check_type": request.check_type,
            "result": result.model_dump(),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error in manual safety check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GDPR/HIPAA Compliance Endpoints ───────────────────────

@app.get("/users/{user_id}/export")
async def export_user_data(user_id: str, request: Request):
    """
    Export all user data (GDPR Article 20 — Right to Data Portability).
    Returns a complete JSON package of the user's data across all collections.
    """
    try:
        ip_address = ""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            ip_address = forwarded.split(",")[0].strip()
        elif request.client:
            ip_address = request.client.host

        compliance_svc = get_compliance_service()
        export = compliance_svc.export_user_data(
            user_id, ip_address=ip_address)
        return JSONResponse(content=export.model_dump())
    except Exception as e:
        logger.error(f"Error exporting user data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/users/{user_id}/data")
async def delete_all_user_data(user_id: str, request: Request):
    """
    Cascade delete all user data (GDPR Article 17 — Right to Erasure).
    Removes data from all collections. Audit logs are preserved (legal requirement).
    """
    try:
        ip_address = ""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            ip_address = forwarded.split(",")[0].strip()
        elif request.client:
            ip_address = request.client.host

        compliance_svc = get_compliance_service()
        receipt = compliance_svc.delete_all_user_data(
            user_id, ip_address=ip_address)
        return JSONResponse(content=receipt.model_dump())
    except Exception as e:
        logger.error(f"Error deleting user data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users/{user_id}/consent")
async def record_consent(user_id: str, request_body: ConsentRequest, request: Request):
    """Record initial consent preferences for a user."""
    try:
        ip_address = ""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            ip_address = forwarded.split(",")[0].strip()
        elif request.client:
            ip_address = request.client.host

        compliance_svc = get_compliance_service()
        record = compliance_svc.record_consent(
            user_id=user_id,
            consents=request_body.consents,
            privacy_policy_version=request_body.privacy_policy_version,
            ip_address=ip_address,
        )
        return JSONResponse(content=record.model_dump())
    except Exception as e:
        logger.error(f"Error recording consent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/consent")
async def get_consent(user_id: str):
    """Get current consent status for a user."""
    try:
        compliance_svc = get_compliance_service()
        record = compliance_svc.get_consent(user_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail="No consent record found")
        return JSONResponse(content=record.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching consent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/users/{user_id}/consent")
async def update_consent(user_id: str, request_body: ConsentRequest, request: Request):
    """Update consent preferences for a user."""
    try:
        ip_address = ""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            ip_address = forwarded.split(",")[0].strip()
        elif request.client:
            ip_address = request.client.host

        compliance_svc = get_compliance_service()
        record = compliance_svc.update_consent(
            user_id=user_id,
            consents=request_body.consents,
            privacy_policy_version=request_body.privacy_policy_version,
            ip_address=ip_address,
        )
        if record is None:
            raise HTTPException(
                status_code=404,
                detail="No existing consent record. Use POST to create one first.",
            )
        return JSONResponse(content=record.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating consent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/retention/enforce", dependencies=[Depends(require_admin)])
async def enforce_retention():
    """
    Trigger retention policy enforcement (admin endpoint).
    Purges stale data per configured retention windows.
    """
    try:
        compliance_svc = get_compliance_service()
        results = compliance_svc.enforce_retention_policies()
        return JSONResponse(content={
            "status": "success",
            "purge_results": results,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error enforcing retention policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/retention/policies", dependencies=[Depends(require_admin)])
async def get_retention_policies():
    """View retention policy configuration."""
    try:
        compliance_svc = get_compliance_service()
        policies = compliance_svc.get_retention_policies()
        return JSONResponse(content={
            "policies": policies,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error fetching retention policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/audit-log", dependencies=[Depends(require_admin)])
async def query_audit_log(
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    limit: int = 100,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Query audit logs with optional filters (admin endpoint).
    Supports filtering by user_id, action, resource_type, date range.
    """
    try:
        audit_svc = get_audit_service()
        logs = audit_svc.query_logs(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )
        return JSONResponse(content={
            "logs": logs,
            "count": len(logs),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error querying audit log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Soundscape Audio Endpoints ─────────────────────────────


@app.get("/soundscapes/{soundscape_id}/audio")
async def get_soundscape_audio_url(soundscape_id: str):
    """
    Get audio for a soundscape.

    Tries GCS signed URL first. Falls back to local file serving
    when LOCAL_AUDIO_FALLBACK is enabled (for development/demo).
    """
    ss_svc = get_soundscape_service()
    ss = ss_svc.get_soundscape(soundscape_id)
    if ss is None:
        raise HTTPException(
            status_code=404,
            detail=f"Soundscape '{soundscape_id}' not found.",
        )

    storage_svc = get_storage_service()

    # Try GCS first (production path)
    if storage_svc.is_configured:
        audio_url = storage_svc.get_audio_url(soundscape_id)
        if audio_url:
            return JSONResponse(content={
                "soundscape_id": soundscape_id,
                "title": ss.title,
                "audio_url": audio_url,
                "source": "gcs",
                "expires_in_minutes": get_config().GCS_SIGNED_URL_EXPIRY_MINUTES,
                "timestamp": datetime.now().isoformat(),
            })

    # Fall back to local audio files (development/demo)
    local_path = storage_svc.get_local_audio_path(soundscape_id)
    if local_path:
        media_type = "audio/mpeg" if local_path.endswith(
            ".mp3") else "audio/wav"
        return FileResponse(
            path=local_path,
            media_type=media_type,
            filename=f"{soundscape_id}{'.mp3' if local_path.endswith('.mp3') else '.wav'}",
        )

    # Neither GCS nor local available
    if not storage_svc.is_configured and not storage_svc.has_local_fallback:
        raise HTTPException(
            status_code=503,
            detail="Audio storage is not configured. Set GCS_AUDIO_BUCKET for production "
                   "or LOCAL_AUDIO_FALLBACK=true with LOCAL_AUDIO_DIR for development.",
        )
    raise HTTPException(
        status_code=404,
        detail=f"No audio file found for soundscape '{soundscape_id}'.",
    )


@app.get("/admin/audio-status", dependencies=[Depends(require_admin)])
async def get_audio_status():
    """
    Admin endpoint: check which soundscapes have audio files uploaded to GCS.
    Returns a dict of {soundscape_id: has_audio} for all 17 soundscapes.
    """
    try:
        ss_svc = get_soundscape_service()
        status = ss_svc.get_audio_status()
        total = len(status)
        uploaded = sum(1 for v in status.values() if v)

        storage_svc = get_storage_service()

        return JSONResponse(content={
            "audio_status": status,
            "summary": {
                "total_soundscapes": total,
                "with_audio": uploaded,
                "without_audio": total - uploaded,
                "gcs_configured": storage_svc.is_configured,
                "bucket": get_config().GCS_AUDIO_BUCKET or "(not configured)",
            },
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error checking audio status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Voice Chat Endpoints ───────────────────────────────────


@app.post("/chat/voice", response_model=VoiceChatResponse)
async def chat_voice(request: VoiceChatRequest, user: dict = Depends(get_current_user)):
    """
    Voice-enabled chat endpoint.

    Auth: same as /chat — Firebase ID token or INTERNAL_SERVICE_KEY.

    Accepts optional base64-encoded audio input, transcribes it to text,
    runs the full chat pipeline via internal call to /chat, then
    optionally converts the response to speech.

    response_format controls output:
      - "text": text response only (no TTS)
      - "audio": audio response only
      - "both": text + audio response (default)
    """
    try:
        # Enforce user_id matches authenticated user (admin/service account exempt)
        if not _is_admin(user):
            if request.user_id and request.user_id != user.get("uid"):
                raise HTTPException(
                    status_code=403,
                    detail="user_id in request does not match authenticated user",
                )
            if not request.user_id:
                request.user_id = user.get("uid")

        voice_svc = get_voice_service()

        # ── Step 1: Resolve input text ──
        transcribed_text = None
        prompt = request.message

        if request.audio_input:
            # Decode base64 audio
            audio_bytes, decode_err = voice_svc.decode_base64_audio(
                request.audio_input
            )
            if decode_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio decode error: {decode_err}",
                )

            # Transcribe
            if not voice_svc.stt_available:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Speech-to-text is not available. "
                        "Install SpeechRecognition: pip install SpeechRecognition"
                    ),
                )

            text, stt_err = voice_svc.transcribe(
                audio_bytes, request.audio_format
            )
            if stt_err:
                raise HTTPException(
                    status_code=422,
                    detail=f"Transcription failed: {stt_err}",
                )

            transcribed_text = text
            prompt = text

        # Validate we have input text
        if not prompt or not prompt.strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    "No input provided. Send either 'message' (text) "
                    "or 'audio_input' (base64 audio)."
                ),
            )

        # ── Step 2: Run full chat pipeline via internal ASGI call ──
        chat_payload = {
            "message": prompt,
            "session_id": request.session_id,
        }
        if request.user_id:
            chat_payload["user_id"] = request.user_id

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://internal",
        ) as client:
            chat_response = await client.post(
                "/chat",
                json=chat_payload,
                timeout=60.0,
            )

        if chat_response.status_code != 200:
            raise HTTPException(
                status_code=chat_response.status_code,
                detail=chat_response.json().get(
                    "message", "Chat pipeline error"
                ),
            )

        chat_data = chat_response.json()

        # ── Step 3: TTS on response (if requested) ──
        audio_output = None
        audio_duration_ms = None

        if request.response_format in ("audio", "both"):
            response_text = chat_data.get("response", "")
            if response_text and voice_svc.tts_available:
                mp3_bytes, tts_err = await voice_svc.synthesize(
                    response_text,
                    voice=request.tts_voice,
                )
                if tts_err:
                    logger.warning(f"TTS synthesis failed: {tts_err}")
                else:
                    audio_output = voice_svc.encode_audio_base64(mp3_bytes)
                    audio_duration_ms = voice_svc.estimate_audio_duration_ms(
                        mp3_bytes
                    )

        # ── Step 4: Build response ──
        return VoiceChatResponse(
            session_id=chat_data.get("session_id", request.session_id),
            response=chat_data.get("response", ""),
            conversation=chat_data.get("conversation", []),
            status=chat_data.get("status", "success"),
            timestamp=chat_data.get(
                "timestamp", datetime.now().isoformat()
            ),
            message_count=chat_data.get("message_count", 0),
            user_id=chat_data.get("user_id"),
            detected_emotion=chat_data.get("detected_emotion"),
            detected_intent=chat_data.get("detected_intent"),
            transcribed_text=transcribed_text,
            audio_output=audio_output,
            audio_duration_ms=audio_duration_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Simple text-to-speech utility endpoint.
    Converts text to base64-encoded MP3 audio.
    """
    try:
        voice_svc = get_voice_service()

        if not voice_svc.tts_available:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Text-to-speech is not available. "
                    "Install edge-tts: pip install edge-tts"
                ),
            )

        mp3_bytes, tts_err = await voice_svc.synthesize(
            request.text,
            voice=request.voice,
            rate=request.rate,
        )

        if tts_err:
            raise HTTPException(
                status_code=422, detail=f"TTS failed: {tts_err}"
            )

        config = get_config()
        return TTSResponse(
            audio=voice_svc.encode_audio_base64(mp3_bytes),
            duration_ms=voice_svc.estimate_audio_duration_ms(mp3_bytes),
            voice=request.voice or config.TTS_VOICE,
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stt", response_model=STTResponse)
async def speech_to_text(request: STTRequest):
    """
    Simple speech-to-text utility endpoint.
    Converts base64-encoded audio to text.
    """
    try:
        voice_svc = get_voice_service()

        if not voice_svc.stt_available:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Speech-to-text is not available. "
                    "Install SpeechRecognition: pip install SpeechRecognition"
                ),
            )

        # Decode base64
        audio_bytes, decode_err = voice_svc.decode_base64_audio(
            request.audio
        )
        if decode_err:
            raise HTTPException(
                status_code=400,
                detail=f"Audio decode error: {decode_err}",
            )

        # Transcribe
        text, stt_err = voice_svc.transcribe(
            audio_bytes, request.audio_format
        )
        if stt_err:
            raise HTTPException(
                status_code=422, detail=f"Transcription failed: {stt_err}"
            )

        return STTResponse(
            text=text,
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 16: Wearable Integration Endpoints ─────────────


@app.post("/wearables/{user_id}/sync")
async def sync_health_data(user_id: str, request: HealthSyncRequest):
    """
    Batch sync daily health metrics from wearable/client.

    The Flutter client collects health data from HealthKit/Health Connect
    and sends it here as structured JSON. Idempotent — safe to re-send.
    """
    try:
        if user_id != request.user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id in path must match user_id in request body",
            )

        wearable_svc = get_wearable_service()

        if not wearable_svc.is_available:
            raise HTTPException(
                status_code=503,
                detail="Wearable service is not available (database not configured)",
            )

        saved, error = wearable_svc.sync_health_data(request)
        if error:
            raise HTTPException(status_code=422, detail=error)

        return {
            "status": "success",
            "records_saved": saved,
            "total_submitted": len(request.metrics),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/wearables/{user_id}/metrics")
async def get_health_metrics(user_id: str, days: int = 7):
    """
    Get recent daily health metrics for a user.

    Returns last N days of sleep + activity data, ordered by date descending.
    """
    try:
        wearable_svc = get_wearable_service()

        if not wearable_svc.is_available:
            raise HTTPException(
                status_code=503,
                detail="Wearable service is not available",
            )

        metrics = wearable_svc.get_recent_metrics(user_id, days=days)

        return {
            "status": "success",
            "user_id": user_id,
            "period_days": days,
            "total_records": len(metrics),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/wearables/{user_id}/summary", response_model=HealthSummary)
async def get_health_summary_endpoint(user_id: str, days: int = 7):
    """
    Get aggregated health summary with trends.

    Compares recent vs previous period to detect improving/declining/stable trends.
    """
    try:
        wearable_svc = get_wearable_service()

        if not wearable_svc.is_available:
            raise HTTPException(
                status_code=503,
                detail="Wearable service is not available",
            )

        summary = wearable_svc.get_health_summary(user_id, days=days)
        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/wearables/{user_id}/sleep-insights")
async def get_sleep_insights_endpoint(user_id: str, days: int = 7):
    """
    Get sleep-specific analysis.

    Returns avg duration, quality distribution, best/worst nights,
    and sleep stage averages.
    """
    try:
        wearable_svc = get_wearable_service()

        if not wearable_svc.is_available:
            raise HTTPException(
                status_code=503,
                detail="Wearable service is not available",
            )

        insights = wearable_svc.get_sleep_insights(user_id, days=days)
        return insights

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sleep insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 17: RL Diagnostic Endpoint ─────────────────


@app.get("/rl/{user_id}/bandit-state")
async def get_bandit_state(user_id: str, emotion: str = "neutral"):
    """
    Get Thompson Sampling bandit arm states for a user (diagnostic).

    Returns arm statistics for the given emotion context, including
    alpha/beta parameters, mean success rate, and total observations.
    """
    try:
        config = get_config()
        if not config.RL_ENABLED:
            return {
                "user_id": user_id,
                "rl_enabled": False,
                "message": "RL is disabled",
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

        rl_svc = get_rl_service()
        arm_stats = rl_svc.get_arm_stats(user_id, emotion)
        emotion_group = rl_svc.get_emotion_group(emotion)

        return {
            "user_id": user_id,
            "rl_enabled": True,
            "emotion": emotion,
            "emotion_group": emotion_group,
            "arms": arm_stats,
            "total_arms": len(arm_stats),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Bandit state error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 18: Fine-Tuning Management Endpoints ──────────


@app.get("/finetuning/status", dependencies=[Depends(require_admin)])
async def get_finetuning_status():
    """
    Get overall fine-tuning feature status including config,
    active model, and current A/B split.
    """
    config = get_config()
    return {
        "ft_enabled": config.FT_ENABLED,
        "ft_model_id": config.FT_MODEL_ID or "(none)",
        "base_model": config.FT_BASE_MODEL,
        "ab_split_percent": config.FT_AB_SPLIT_PERCENT,
        "min_training_examples": config.FT_MIN_TRAINING_EXAMPLES,
        "min_feedback_rating": config.FT_MIN_FEEDBACK_RATING,
        "min_helpfulness_score": config.FT_MIN_HELPFULNESS_SCORE,
        "training_epochs": config.FT_TRAINING_EPOCHS,
        "status": "success",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/finetuning/extract-training-data", dependencies=[Depends(require_admin)])
async def extract_training_data(request: ExtractTrainingDataRequest):
    """
    Extract high-quality training examples from chat sessions.
    Only includes data from users who have consented to ML_TRAINING.
    Filters by feedback rating and helpfulness scores.
    """
    try:
        config = get_config()
        if not config.FT_ENABLED:
            return {
                "ft_enabled": False,
                "message": "Fine-tuning is disabled",
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

        ft_svc = get_finetuning_service()
        examples = ft_svc.extract_training_data(
            min_feedback_rating=request.min_feedback_rating,
            min_helpfulness_score=request.min_helpfulness_score,
            max_examples=request.max_examples,
        )

        return {
            "examples_extracted": len(examples),
            "min_feedback_rating": request.min_feedback_rating,
            "min_helpfulness_score": request.min_helpfulness_score,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Training data extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/finetuning/submit-job", dependencies=[Depends(require_admin)])
async def submit_finetuning_job(request: SubmitFineTuningJobRequest):
    """
    Submit a fine-tuning job to OpenAI using extracted training data.
    Formats data as JSONL, uploads to OpenAI Files API, creates the job.
    """
    try:
        config = get_config()
        if not config.FT_ENABLED:
            return {
                "ft_enabled": False,
                "message": "Fine-tuning is disabled",
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

        ft_svc = get_finetuning_service()

        # Fetch unused training examples from Firestore
        if ft_svc._db is None:
            raise HTTPException(
                status_code=503, detail="Database not available"
            )

        unused_examples = []
        docs = (
            ft_svc._db.collection(ft_svc.EXAMPLES_COLLECTION)
            .where("included_in_job", "==", "")
            .limit(config.FT_MIN_TRAINING_EXAMPLES * 10)
            .stream()
        )
        from models import TrainingExample
        for doc in docs:
            data = doc.to_dict()
            unused_examples.append(TrainingExample(**data))

        if len(unused_examples) < config.FT_MIN_TRAINING_EXAMPLES:
            return {
                "message": (
                    f"Not enough training examples: "
                    f"{len(unused_examples)} < "
                    f"{config.FT_MIN_TRAINING_EXAMPLES} minimum"
                ),
                "examples_available": len(unused_examples),
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }

        record = ft_svc.submit_finetuning_job(
            examples=unused_examples,
            base_model=request.base_model,
            n_epochs=request.n_epochs,
            suffix=request.suffix,
        )

        if record is None:
            raise HTTPException(
                status_code=500, detail="Job submission failed"
            )

        return {
            "job": record.model_dump(),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fine-tuning job submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/finetuning/jobs", dependencies=[Depends(require_admin)])
async def list_finetuning_jobs(limit: int = 10):
    """List all fine-tuning jobs, most recent first."""
    try:
        ft_svc = get_finetuning_service()
        jobs = ft_svc.list_jobs(limit=limit)
        return {
            "jobs": jobs,
            "total": len(jobs),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"List jobs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/finetuning/jobs/{job_id}", dependencies=[Depends(require_admin)])
async def get_finetuning_job(job_id: str):
    """
    Get status of a specific fine-tuning job.
    Fetches latest status from OpenAI API and updates local record.
    """
    try:
        ft_svc = get_finetuning_service()
        record = ft_svc.check_job_status(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "job": record.model_dump(),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/finetuning/ab-stats", dependencies=[Depends(require_admin)])
async def get_ab_stats():
    """
    Get A/B testing comparison stats between base and fine-tuned models.
    """
    try:
        ft_svc = get_finetuning_service()
        stats = ft_svc.compute_ab_stats()
        return stats.model_dump()
    except Exception as e:
        logger.error(f"A/B stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Monitoring Dashboard (Phase 19) ─────────────────────────


@app.get("/admin/dashboard", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
async def dashboard_page():
    """Admin monitoring dashboard HTML page with auto-refresh and Chart.js charts."""
    config = get_config()
    if not config.DASHBOARD_ENABLED:
        return HTMLResponse(
            content="<h1>Dashboard disabled</h1><p>Set DASHBOARD_ENABLED=true to enable.</p>",
            status_code=403,
        )
    refresh_interval = config.DASHBOARD_REFRESH_INTERVAL
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LucilleLLM - Monitoring Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;min-height:100vh}
header{background:#0f3460;padding:16px 24px;display:flex;justify-content:space-between;align-items:center;box-shadow:0 2px 8px rgba(0,0,0,.3)}
header h1{font-size:1.4em;color:#53c0f5;font-weight:600}
#status{font-size:.85em;color:#8892a0}
#status span{margin-left:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px;padding:20px;max-width:1600px;margin:0 auto}
.card{background:#16213e;border-radius:12px;padding:20px;box-shadow:0 2px 12px rgba(0,0,0,.2);transition:box-shadow .2s}
.card:hover{box-shadow:0 4px 20px rgba(0,0,0,.4)}
.card h2{color:#53c0f5;font-size:1.1em;border-bottom:2px solid #0f3460;padding-bottom:8px;margin-bottom:14px}
.metrics-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.metric{text-align:center;padding:8px}
.metric-value{font-size:1.8em;font-weight:bold;color:#e94560;line-height:1.2}
.metric-label{font-size:.75em;color:#8892a0;text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
.status-ok{color:#4ade80}.status-warn{color:#fbbf24}.status-error{color:#ef4444}
.chart-container{position:relative;height:200px;margin-top:12px}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:.82em}
th{text-align:left;color:#53c0f5;padding:6px 8px;border-bottom:2px solid #0f3460}
td{padding:6px 8px;border-bottom:1px solid #0f346044;color:#c0c0c0}
tr:hover td{background:#0f346033}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.75em;font-weight:600}
.badge-critical{background:#ef444433;color:#ef4444}
.badge-high{background:#f9731633;color:#f97316}
.badge-moderate{background:#fbbf2433;color:#fbbf24}
.badge-low{background:#4ade8033;color:#4ade80}
.rl-group{margin-top:8px}
.rl-group-title{color:#e94560;font-size:.9em;font-weight:600;text-transform:capitalize;margin-bottom:4px}
.rl-arm{display:flex;justify-content:space-between;padding:3px 8px;font-size:.82em;background:#0f346022;border-radius:4px;margin-bottom:3px}
.rl-arm span:first-child{color:#c0c0c0}.rl-arm span:last-child{color:#4ade80;font-weight:600}
.spinner{display:inline-block;border:3px solid #16213e;border-top:3px solid #e94560;border-radius:50%;width:20px;height:20px;animation:spin .8s linear infinite;margin:40px auto;display:block}
@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
.empty-msg{color:#8892a0;font-style:italic;text-align:center;padding:20px;font-size:.9em}
</style>
</head>
<body>
<header>
<h1>LucilleLLM Monitoring Dashboard</h1>
<div id="status">
<span id="lastUpdate">Loading...</span>
<span id="refreshCountdown"></span>
</div>
</header>
<main>
<div class="grid">
<div class="card" id="systemCard">
<h2>System Health</h2>
<div class="metrics-grid" id="systemMetrics"><div class="spinner"></div></div>
<div class="chart-container"><canvas id="latencyChart"></canvas></div>
</div>
<div class="card" id="engagementCard">
<h2>User Engagement</h2>
<div class="metrics-grid" id="engagementMetrics"><div class="spinner"></div></div>
</div>
<div class="card" id="therapyCard">
<h2>Therapy Effectiveness</h2>
<div class="metrics-grid" id="therapyMetrics"><div class="spinner"></div></div>
<div style="display:flex;gap:12px;margin-top:12px">
<div class="chart-container" style="flex:1"><canvas id="modalityChart"></canvas></div>
<div class="chart-container" style="flex:1"><canvas id="moodChart"></canvas></div>
</div>
</div>
<div class="card" id="safetyCard">
<h2>Safety Overview</h2>
<div class="metrics-grid" id="safetyMetrics"><div class="spinner"></div></div>
<div style="display:flex;gap:12px;margin-top:12px">
<div class="chart-container" style="flex:1;max-width:200px"><canvas id="riskChart"></canvas></div>
<div style="flex:2;overflow-x:auto" id="safetyTableWrap"></div>
</div>
</div>
<div class="card" id="modelsCard">
<h2>Model Performance (A/B)</h2>
<div class="metrics-grid" id="modelMetrics"><div class="spinner"></div></div>
<div class="chart-container"><canvas id="abChart"></canvas></div>
</div>
<div class="card" id="rlCard">
<h2>RL Bandit Overview</h2>
<div class="metrics-grid" id="rlMetrics"><div class="spinner"></div></div>
<div id="rlArms"></div>
</div>
</div>
</main>
<script>
const REFRESH_MS=REFRESH_INTERVAL_PLACEHOLDER*1000;
let charts={};
function m(id,val,label){return`<div class="metric"><div class="metric-value">${val}</div><div class="metric-label">${label}</div></div>`}
function fmtUp(s){const h=Math.floor(s/3600),mn=Math.floor((s%3600)/60),sc=Math.floor(s%60);return`${h}h ${mn}m ${sc}s`}
function errClass(r){return r<0.01?'status-ok':r<0.05?'status-warn':'status-error'}
function pct(v){return(v*100).toFixed(1)+'%'}

function updateSystem(d){
 const el=document.getElementById('systemMetrics');
 el.innerHTML=m('up',fmtUp(d.uptime_seconds),'Uptime')
  +m('req',d.total_requests.toLocaleString(),'Total Requests')
  +m('err',`<span class="${errClass(d.overall_error_rate)}">${pct(d.overall_error_rate)}</span>`,'Error Rate')
  +m('p50',d.latency_p50.toFixed(0)+'ms','Latency P50')
  +m('p95',d.latency_p95.toFixed(0)+'ms','Latency P95')
  +m('cache',pct(d.cache_hit_rate),'Cache Hit Rate')
  +m('mem',d.memory_rss_mb+'MB','Memory RSS')
  +m('env',d.environment,'Environment')
  +m('mod',d.model,'Model');
 // Latency bar chart
 if(d.top_endpoints&&d.top_endpoints.length>0){
  if(charts.lat)charts.lat.destroy();
  const ctx=document.getElementById('latencyChart').getContext('2d');
  charts.lat=new Chart(ctx,{type:'bar',data:{
   labels:d.top_endpoints.map(e=>e.path.length>25?e.path.slice(0,25)+'...':e.path),
   datasets:[{label:'P50 ms',data:d.top_endpoints.map(e=>e.latency_p50),backgroundColor:'#e9456088',borderColor:'#e94560',borderWidth:1},
    {label:'P95 ms',data:d.top_endpoints.map(e=>e.latency_p95||0),backgroundColor:'#53c0f588',borderColor:'#53c0f5',borderWidth:1}]
  },options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',
   scales:{x:{ticks:{color:'#8892a0'},grid:{color:'#0f346044'}},y:{ticks:{color:'#8892a0',font:{size:10}},grid:{display:false}}},
   plugins:{legend:{labels:{color:'#8892a0',font:{size:10}}}}}});
 }
}

function updateEngagement(d){
 document.getElementById('engagementMetrics').innerHTML=
  m('tu',d.total_users.toLocaleString(),'Total Users')
  +m('au',d.active_users_today.toLocaleString(),'Active Today')
  +m('ts',d.total_sessions.toLocaleString(),'Total Sessions')
  +m('mt',d.total_messages_today.toLocaleString(),'Messages Today')
  +m('nu',d.new_users_7d.toLocaleString(),'New Users (7d)')
  +m('as',d.avg_sessions_per_user.toFixed(1),'Avg Sessions/User');
}

function updateTherapy(d){
 document.getElementById('therapyMetrics').innerHTML=
  m('to',d.total_exercise_outcomes.toLocaleString(),'Total Outcomes')
  +m('ah',d.avg_helpfulness.toFixed(1)+'/5','Avg Helpfulness')
  +m('cr',pct(d.exercise_completion_rate),'Completion Rate')
  +m('am',(d.avg_mood_improvement>=0?'+':'')+d.avg_mood_improvement.toFixed(1),'Avg Mood Change');
 // Modality bar chart
 const mods=d.modality_breakdown||{};
 const modKeys=Object.keys(mods);
 if(modKeys.length>0){
  if(charts.mod)charts.mod.destroy();
  charts.mod=new Chart(document.getElementById('modalityChart').getContext('2d'),{
   type:'bar',data:{labels:modKeys,datasets:[{label:'Avg Helpfulness',
    data:modKeys.map(k=>mods[k].avg_helpfulness),backgroundColor:'#4ade8088',borderColor:'#4ade80',borderWidth:1}]},
   options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',
    scales:{x:{min:0,max:5,ticks:{color:'#8892a0'},grid:{color:'#0f346044'}},y:{ticks:{color:'#8892a0'},grid:{display:false}}},
    plugins:{legend:{display:false}}}});
 }
 // Mood distribution doughnut
 const md=d.mood_improvement_distribution||{};
 if(md.improved||md.unchanged||md.worsened){
  if(charts.mood)charts.mood.destroy();
  charts.mood=new Chart(document.getElementById('moodChart').getContext('2d'),{
   type:'doughnut',data:{labels:['Improved','Unchanged','Worsened'],
    datasets:[{data:[md.improved||0,md.unchanged||0,md.worsened||0],
     backgroundColor:['#4ade80','#fbbf24','#ef4444'],borderWidth:0}]},
   options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8892a0',font:{size:10}}}}}});
 }
}

function updateSafety(d){
 const el=document.getElementById('safetyMetrics');
 el.innerHTML=m('te',d.total_events.toLocaleString(),'Total Events');
 // Risk level doughnut
 const rl=d.events_by_risk_level||{};
 const rlKeys=Object.keys(rl);
 if(rlKeys.length>0){
  if(charts.risk)charts.risk.destroy();
  const colors={'critical':'#ef4444','high':'#f97316','moderate':'#fbbf24','low':'#4ade80','CRITICAL':'#ef4444','HIGH':'#f97316','MODERATE':'#fbbf24','LOW':'#4ade80'};
  charts.risk=new Chart(document.getElementById('riskChart').getContext('2d'),{
   type:'doughnut',data:{labels:rlKeys,datasets:[{data:rlKeys.map(k=>rl[k]),
    backgroundColor:rlKeys.map(k=>colors[k]||'#8892a0'),borderWidth:0}]},
   options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8892a0',font:{size:10}}}}}});
 }
 // Recent events table
 const wrap=document.getElementById('safetyTableWrap');
 const evts=d.recent_critical_events||[];
 if(evts.length===0){wrap.innerHTML='<p class="empty-msg">No critical events</p>';return}
 let html='<table><tr><th>Risk</th><th>Type</th><th>Action</th><th>Time</th></tr>';
 evts.forEach(e=>{
  const bc=e.risk_level.toLowerCase().includes('critical')?'badge-critical':'badge-high';
  const t=e.created_at?new Date(e.created_at).toLocaleString():'';
  html+=`<tr><td><span class="badge ${bc}">${e.risk_level}</span></td><td>${e.event_type}</td><td>${e.action_taken}</td><td>${t}</td></tr>`;
 });
 wrap.innerHTML=html+'</table>';
}

function updateModels(d){
 const st=d.ft_enabled?'<span class="status-ok">Enabled</span>':'<span class="status-warn">Disabled</span>';
 document.getElementById('modelMetrics').innerHTML=
  m('ft',st,'Fine-Tuning')
  +m('mid',d.active_model_id||'N/A','Active FT Model')
  +m('sp',d.ab_split_percent+'%','A/B Split');
 // A/B comparison bar chart
 if(d.base_total>0||d.ft_total>0){
  if(charts.ab)charts.ab.destroy();
  charts.ab=new Chart(document.getElementById('abChart').getContext('2d'),{
   type:'bar',data:{labels:['Helpful Rate','Avg Length (chars)'],
    datasets:[
     {label:'Base (n='+d.base_total+')',data:[d.base_helpful_rate*100,d.base_avg_length],backgroundColor:'#53c0f588',borderColor:'#53c0f5',borderWidth:1},
     {label:'Fine-Tuned (n='+d.ft_total+')',data:[d.ft_helpful_rate*100,d.ft_avg_length],backgroundColor:'#e9456088',borderColor:'#e94560',borderWidth:1}
    ]},
   options:{responsive:true,maintainAspectRatio:false,
    scales:{x:{ticks:{color:'#8892a0'},grid:{display:false}},y:{ticks:{color:'#8892a0'},grid:{color:'#0f346044'}}},
    plugins:{legend:{labels:{color:'#8892a0',font:{size:10}}}}}});
 }
}

function updateRL(d){
 const st=d.rl_enabled?'<span class="status-ok">Enabled</span>':'<span class="status-warn">Disabled</span>';
 document.getElementById('rlMetrics').innerHTML=
  m('rl',st,'RL Status')
  +m('us',d.total_users_with_bandit_state.toLocaleString(),'Users w/ State')
  +m('ar',d.total_arms.toLocaleString(),'Total Arms');
 const el=document.getElementById('rlArms');
 const groups=d.top_arms_by_group||{};
 const gKeys=Object.keys(groups);
 if(gKeys.length===0){el.innerHTML='<p class="empty-msg">No bandit data</p>';return}
 let html='';
 gKeys.forEach(g=>{
  html+=`<div class="rl-group"><div class="rl-group-title">${g}</div>`;
  (groups[g]||[]).forEach(a=>{
   html+=`<div class="rl-arm"><span>${a.exercise_id}</span><span>${(a.mean_success_rate*100).toFixed(0)}% (n=${a.total_users})</span></div>`;
  });
  html+='</div>';
 });
 el.innerHTML=html;
}

async function fetchAll(){
 try{
  const r=await fetch('/admin/dashboard/all');
  if(!r.ok)throw new Error('HTTP '+r.status);
  const d=await r.json();
  updateSystem(d.system);
  updateEngagement(d.engagement);
  updateTherapy(d.therapy);
  updateSafety(d.safety);
  updateModels(d.models);
  updateRL(d.rl);
  document.getElementById('lastUpdate').textContent='Last update: '+new Date().toLocaleTimeString();
 }catch(e){
  console.error('Dashboard fetch error:',e);
  document.getElementById('lastUpdate').textContent='Error: '+e.message;
 }
}

fetchAll();
setInterval(fetchAll,REFRESH_MS);
let cd=REFRESH_MS/1000;
setInterval(()=>{cd--;if(cd<=0)cd=REFRESH_MS/1000;
 document.getElementById('refreshCountdown').textContent='(refresh in '+cd+'s)';},1000);
</script>
</body>
</html>"""
    html_content = html_content.replace(
        "REFRESH_INTERVAL_PLACEHOLDER", str(refresh_interval)
    )
    return HTMLResponse(content=html_content)


@app.get("/admin/dashboard/system", dependencies=[Depends(require_admin)])
async def dashboard_system_metrics():
    """System health metrics for the monitoring dashboard."""
    try:
        svc = get_monitoring_service()
        return svc.get_system_metrics().model_dump()
    except Exception as e:
        logger.error(f"Dashboard system metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/dashboard/engagement", dependencies=[Depends(require_admin)])
async def dashboard_engagement_metrics():
    """User engagement metrics for the monitoring dashboard."""
    try:
        svc = get_monitoring_service()
        return svc.get_engagement_metrics().model_dump()
    except Exception as e:
        logger.error(f"Dashboard engagement metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/dashboard/therapy", dependencies=[Depends(require_admin)])
async def dashboard_therapy_metrics():
    """Therapy effectiveness metrics for the monitoring dashboard."""
    try:
        svc = get_monitoring_service()
        return svc.get_therapy_metrics().model_dump()
    except Exception as e:
        logger.error(f"Dashboard therapy metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/dashboard/safety", dependencies=[Depends(require_admin)])
async def dashboard_safety_metrics():
    """Safety overview metrics for the monitoring dashboard."""
    try:
        svc = get_monitoring_service()
        return svc.get_safety_metrics().model_dump()
    except Exception as e:
        logger.error(f"Dashboard safety metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/dashboard/models", dependencies=[Depends(require_admin)])
async def dashboard_model_metrics():
    """A/B model performance metrics for the monitoring dashboard."""
    try:
        svc = get_monitoring_service()
        return svc.get_model_metrics().model_dump()
    except Exception as e:
        logger.error(f"Dashboard model metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/dashboard/rl", dependencies=[Depends(require_admin)])
async def dashboard_rl_metrics():
    """RL bandit overview metrics for the monitoring dashboard."""
    try:
        svc = get_monitoring_service()
        return svc.get_rl_metrics().model_dump()
    except Exception as e:
        logger.error(f"Dashboard RL metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/dashboard/all", dependencies=[Depends(require_admin)])
async def dashboard_all_metrics():
    """All dashboard metrics combined in a single response."""
    try:
        svc = get_monitoring_service()
        return svc.get_all_metrics().model_dump()
    except Exception as e:
        logger.error(f"Dashboard all metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# Phase 20: Human Escalation & Annual Review Endpoints
# ═══════════════════════════════════════════════════════════


@app.get("/admin/escalations", dependencies=[Depends(require_admin)])
async def list_escalations(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 50,
):
    """List escalation tickets with optional filters."""
    try:
        svc = get_escalation_service()
        escalations = svc.list_escalations(
            status=status, priority=priority, limit=limit)
        return JSONResponse(content={"status": "success", "escalations": escalations, "count": len(escalations)})
    except Exception as e:
        logger.error(f"List escalations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/escalations/stats", dependencies=[Depends(require_admin)])
async def get_escalation_stats():
    """Get aggregate statistics for the escalation queue."""
    try:
        svc = get_escalation_service()
        stats = svc.get_escalation_stats()
        return JSONResponse(content={"status": "success", "stats": stats.model_dump()})
    except Exception as e:
        logger.error(f"Escalation stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/escalations/{escalation_id}", dependencies=[Depends(require_admin)])
async def get_escalation(escalation_id: str):
    """Get a single escalation ticket by ID."""
    try:
        svc = get_escalation_service()
        escalation = svc.get_escalation(escalation_id)
        if escalation is None:
            raise HTTPException(status_code=404, detail="Escalation not found")
        return JSONResponse(content={"status": "success", "escalation": escalation})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get escalation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/admin/escalations/{escalation_id}", dependencies=[Depends(require_admin)])
async def update_escalation(escalation_id: str, update: UpdateEscalationRequest):
    """Update an escalation ticket (status, notes, resolved_by)."""
    try:
        svc = get_escalation_service()
        updated = svc.update_escalation(escalation_id, update)
        if updated is None:
            raise HTTPException(status_code=404, detail="Escalation not found")
        return JSONResponse(content={"status": "success", "escalation": updated})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update escalation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reviews/{user_id}/generate")
async def generate_review(user_id: str, request: GenerateReviewRequest):
    """Generate a comprehensive periodic review for a user."""
    try:
        svc = get_escalation_service()
        review = svc.generate_annual_review(
            user_id=user_id,
            period_start=request.period_start,
            period_end=request.period_end,
        )
        if review is None:
            raise HTTPException(
                status_code=503,
                detail="Review generation unavailable (disabled, no DB, or no OpenAI client)",
            )
        return JSONResponse(content={"status": "success", "review": review.model_dump()})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generate review error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reviews/{user_id}")
async def list_reviews(user_id: str):
    """List all reviews for a user, newest first."""
    try:
        svc = get_escalation_service()
        reviews = svc.list_reviews(user_id)
        return JSONResponse(content={"status": "success", "reviews": reviews, "count": len(reviews)})
    except Exception as e:
        logger.error(f"List reviews error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reviews/{user_id}/{review_id}")
async def get_review(user_id: str, review_id: str):
    """Get a specific review by ID."""
    try:
        svc = get_escalation_service()
        review = svc.get_review(user_id, review_id)
        if review is None:
            raise HTTPException(status_code=404, detail="Review not found")
        return JSONResponse(content={"status": "success", "review": review})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get review error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_code": 422,
            "message": "Validation error",
            "details": jsonable_encoder(exc.errors()),
            "timestamp": datetime.now().isoformat()
        }
    )

# ── Assessment System Endpoints (Phase 21) ───────────────


@app.get("/assessments/instruments")
async def list_assessment_instruments():
    """List available validated assessment instruments (PHQ-9, GAD-7, WHO-5)."""
    try:
        svc = get_assessment_service()
        instruments = svc.get_instruments()
        return JSONResponse(content={
            "instruments": instruments,
            "count": len(instruments),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error listing assessment instruments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/assessments/instruments/{assessment_type}")
async def get_instrument_questions(assessment_type: str):
    """Get the questions for a specific validated instrument."""
    try:
        atype = AssessmentType(assessment_type)
        svc = get_assessment_service()
        questions = svc.get_instrument_questions(atype)
        instrument = next(
            i for i in svc.get_instruments()
            if i["assessment_type"] == assessment_type
        )
        return JSONResponse(content={
            "assessment_type": assessment_type,
            "name": instrument["name"],
            "preamble": instrument["preamble"],
            "questions": [q.model_dump() for q in questions],
            "count": len(questions),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid assessment type: {assessment_type}. Valid: phq9, gad7, who5"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting instrument questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assessments/{user_id}/start", response_model=StartAssessmentResponse)
async def start_assessment(user_id: str, request: StartAssessmentRequest):
    """Start a validated mental health assessment for a user."""
    try:
        svc = get_assessment_service()

        # Check for existing in-progress session
        existing = svc.get_in_progress_session(
            user_id, request.assessment_type)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"User already has an in-progress {request.assessment_type.value} assessment "
                    f"(session {existing.session_id}). Complete or abandon it first."
                )
            )

        session = svc.start_assessment(user_id, request.assessment_type)
        if session is None:
            raise HTTPException(
                status_code=503,
                detail="Assessment system is currently disabled"
            )

        questions = svc.get_instrument_questions(request.assessment_type)

        return StartAssessmentResponse(
            session_id=session.session_id,
            assessment_type=session.assessment_type.value,
            total_questions=session.total_questions,
            first_question=questions[0],
            status="success",
            timestamp=datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assessments/{user_id}/answer/{session_id}", response_model=SubmitAnswerResponse)
async def submit_assessment_answer(
    user_id: str, session_id: str, request: SubmitAnswerRequest
):
    """Submit an answer to the current assessment question."""
    try:
        svc = get_assessment_service()
        session, next_question, safety_notice = svc.submit_answer(
            user_id, session_id, request.value
        )

        if session is None:
            raise HTTPException(
                status_code=404, detail="Assessment session not found")

        is_complete = session.current_question_index >= session.total_questions

        return SubmitAnswerResponse(
            session_id=session.session_id,
            question_answered=session.current_question_index - 1,
            next_question=next_question,
            is_complete=is_complete,
            safety_notice=safety_notice,
            status="success",
            timestamp=datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting assessment answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assessments/{user_id}/complete/{session_id}", response_model=CompleteAssessmentResponse)
async def complete_assessment(user_id: str, session_id: str):
    """Complete an assessment and compute scores using published algorithms."""
    try:
        svc = get_assessment_service()
        session = svc.complete_assessment(user_id, session_id)

        if session is None:
            raise HTTPException(
                status_code=404, detail="Assessment session not found")

        if session.result is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Cannot complete: {len(session.answers)}/{session.total_questions} "
                    f"questions answered. Answer all questions first."
                )
            )

        # Include safety resources if any concern flags
        safety_resources = None
        if session.result.concern_flags:
            try:
                safety_svc = get_safety_service()
                resources = safety_svc.get_crisis_resources()
                safety_resources = [r.model_dump() for r in resources]
            except Exception:
                pass

        return CompleteAssessmentResponse(
            session_id=session.session_id,
            result=session.result,
            safety_flagged=session.safety_flagged,
            safety_resources=safety_resources,
            status="success",
            timestamp=datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/assessments/{user_id}/history")
async def get_assessment_history(
    user_id: str,
    assessment_type: Optional[str] = None,
    limit: int = 20,
):
    """Get past completed assessments for a user."""
    try:
        svc = get_assessment_service()

        atype = None
        if assessment_type:
            try:
                atype = AssessmentType(assessment_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid assessment type: {assessment_type}"
                )

        history = svc.get_assessment_history(user_id, atype, limit)

        return JSONResponse(content={
            "user_id": user_id,
            "assessments": history,
            "count": len(history),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assessment history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/assessments/{user_id}/latest")
async def get_latest_assessment_scores(user_id: str):
    """Get the most recent score for each assessment instrument."""
    try:
        svc = get_assessment_service()
        scores = svc.get_latest_scores(user_id)

        return JSONResponse(content={
            "user_id": user_id,
            "scores": scores,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting latest scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/assessments/{user_id}/wellness-score")
async def get_wellness_score(user_id: str):
    """
    Get composite wellness score.

    Primary score = WHO-5 (0-100, higher = better well-being).
    Breakdowns = PHQ-9 (depression) + GAD-7 (anxiety).

    All scores from clinically validated instruments only.
    """
    try:
        svc = get_assessment_service()
        wellness = svc.get_wellness_score(user_id)

        return JSONResponse(content={
            **wellness.model_dump(),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting wellness score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Self-Care Score (Phase 22) ────────────────────────────


@app.get("/users/{user_id}/selfcare-score", dependencies=[Depends(require_same_user())])
async def get_selfcare_score(user_id: str):
    """
    Get composite self-care engagement score (0-100).

    Combines mood stability, exercise engagement, exercise effectiveness,
    task completion, and consistency into a single score with category
    breakdowns and personalized insights.

    This is NOT a clinical assessment — it measures engagement with
    self-care activities, not clinical mental health status.
    """
    config = get_config()
    if not config.SELFCARE_SCORE_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Self-Care Score is currently disabled.",
        )

    try:
        from selfcare_score_service import get_selfcare_score_service
        svc = get_selfcare_score_service()
        result = svc.compute_score(user_id)

        return JSONResponse(content={
            **result.model_dump(),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error computing self-care score for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Exception Handlers ───────────────────────────────────


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return consistent JSON for Pydantic validation errors (422)."""
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error_code": 422,
            "message": "Validation error",
            "details": exc.errors(),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": exc.status_code,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_code": 500,
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# Development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
