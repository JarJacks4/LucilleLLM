"""
LucilleLLM - Configuration Service

Centralized configuration loaded from environment variables with sensible defaults.
Frozen dataclass ensures immutability after initialization.

Follows the singleton pattern from other services.
"""

import os
from dataclasses import dataclass
from typing import Optional

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded from environment variables."""

    # Environment
    ENVIRONMENT: str = "development"

    # Model configuration
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    FAISS_INDEX_PATH: str = "./faiss_vecdb"

    # Rate limiting
    RATE_LIMIT_CHAT: int = 10          # requests per window for /chat endpoints
    RATE_LIMIT_GLOBAL: int = 100       # requests per window global
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # Cache TTLs (seconds)
    CACHE_TTL_USER_PROFILE: int = 60
    CACHE_TTL_EFFECTIVENESS: int = 120
    CACHE_TTL_EMOTION: int = 300
    CACHE_MAX_SIZE: int = 500

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"           # "json" or "text"

    # Metrics
    METRICS_ENABLED: bool = True

    # Dependency Detection Thresholds
    DEPENDENCY_MAX_MESSAGES_PER_HOUR: int = 20
    DEPENDENCY_MAX_SESSIONS_PER_DAY: int = 8
    DEPENDENCY_NIGHTTIME_START_HOUR: int = 0       # midnight
    DEPENDENCY_NIGHTTIME_END_HOUR: int = 5         # 5am
    DEPENDENCY_NIGHTTIME_THRESHOLD: int = 5        # nighttime messages per week
    DEPENDENCY_ESCALATION_RATIO: float = 1.5       # week-over-week message increase ratio
    DEPENDENCY_CONSECUTIVE_DAYS_THRESHOLD: int = 14  # 2+ weeks daily usage
    DEPENDENCY_COOLDOWN_MINUTES: int = 30          # suggested break duration

    # Cultural Competence
    CULTURAL_BIAS_CHECK_ENABLED: bool = True
    CULTURAL_DEFAULT_COUNTRY: str = "US"

    # Data Retention (days) — GDPR/HIPAA Compliance
    RETENTION_CHAT_SESSIONS_DAYS: int = 365           # 1 year
    RETENTION_MEMORIES_DAYS: int = 730                # 2 years
    RETENTION_FEEDBACK_DAYS: int = 365                # 1 year
    RETENTION_EXERCISE_SESSIONS_DAYS: int = 365       # 1 year
    RETENTION_SOUNDSCAPE_SESSIONS_DAYS: int = 180     # 6 months
    RETENTION_SAFETY_EVENTS_DAYS: int = 730           # 2 years (regulatory)
    RETENTION_INTERACTION_METRICS_DAYS: int = 365     # 1 year
    RETENTION_AUDIT_LOG_DAYS: int = 2555              # 7 years (HIPAA)

    # Audit Logging
    AUDIT_LOG_ENABLED: bool = True

    # Audio Storage (GCS)
    GCS_AUDIO_BUCKET: str = ""                  # e.g. "lucille-soundscapes"
    GCS_AUDIO_PREFIX: str = "soundscapes/"      # path prefix in bucket
    GCS_SIGNED_URL_EXPIRY_MINUTES: int = 60     # signed URL TTL (1 hour)

    # Local Audio Fallback (when GCS is not configured)
    LOCAL_AUDIO_FALLBACK: bool = False           # enable local file serving for dev
    LOCAL_AUDIO_DIR: str = ""                    # e.g. "./audio_output"

    # Voice I/O (TTS / STT)
    TTS_PROVIDER: str = "edge-tts"              # "edge-tts", "openai", "google"
    STT_PROVIDER: str = "speech_recognition"    # "speech_recognition", "openai", "google"
    TTS_VOICE: str = "en-US-AriaNeural"         # default edge-tts voice
    TTS_RATE: str = "+0%"                       # speech rate adjustment
    MAX_AUDIO_SIZE_MB: int = 10                 # max audio upload size in MB

    # Wearable Integration
    WEARABLE_SYNC_ENABLED: bool = True
    WEARABLE_HEALTH_CONTEXT_DAYS: int = 7       # days of health data in LLM context
    RETENTION_HEALTH_METRICS_DAYS: int = 365    # 1 year retention

    # RL / Thompson Sampling (Phase 17)
    RL_ENABLED: bool = True                     # master toggle
    RL_EXPLORATION_BONUS: float = 0.5           # max bonus from TS (0-1 sampled * this)
    RL_MIN_OUTCOMES_FOR_RL: int = 3             # min outcomes before RL kicks in
    RL_SUCCESS_THRESHOLD: int = 4               # helpfulness >= this = success
    RL_CACHE_TTL: int = 300                     # bandit state cache TTL (seconds)

    # Fine-Tuning (Phase 18)
    FT_ENABLED: bool = False                    # master toggle (off by default)
    FT_MODEL_ID: str = ""                       # active fine-tuned model ID
    FT_BASE_MODEL: str = "gpt-4o-mini"         # base model for fine-tuning jobs
    FT_AB_SPLIT_PERCENT: int = 50              # % of users routed to FT model (0-100)
    FT_MIN_TRAINING_EXAMPLES: int = 100        # min examples before allowing job submission
    FT_MIN_FEEDBACK_RATING: str = "helpful"    # min feedback rating to include
    FT_MIN_HELPFULNESS_SCORE: int = 4          # min exercise outcome helpfulness (1-5)
    FT_TRAINING_EPOCHS: int = 3               # training epochs
    FT_RETENTION_TRAINING_DATA_DAYS: int = 365 # retention for training data

    # Monitoring Dashboard (Phase 19)
    DASHBOARD_ENABLED: bool = True             # master toggle
    DASHBOARD_CACHE_TTL: int = 60              # cache TTL for aggregation queries (seconds)
    DASHBOARD_REFRESH_INTERVAL: int = 30       # auto-refresh interval for HTML page (seconds)

    # Human Escalation & Reviews (Phase 20)
    ESCALATION_ENABLED: bool = True                    # master toggle for auto-escalation
    ESCALATION_REPEATED_HIGH_THRESHOLD: int = 3        # HIGH events in window to trigger
    ESCALATION_REPEATED_HIGH_WINDOW_DAYS: int = 7      # rolling window for repeated check
    REVIEW_ENABLED: bool = True                        # master toggle for annual reviews
    REVIEW_DEFAULT_PERIOD_DAYS: int = 365              # default review period

    # Assessment System (Phase 21)
    ASSESSMENT_ENABLED: bool = True                    # master toggle
    ASSESSMENT_REMINDER_DAYS: int = 14                 # remind users to reassess every N days
    ASSESSMENT_WHO5_CONCERN_THRESHOLD: int = 50        # WHO-5 scaled score below this = concern
    ASSESSMENT_PHQ9_CONCERN_THRESHOLD: int = 10        # PHQ-9 score at/above this = concern
    ASSESSMENT_GAD7_CONCERN_THRESHOLD: int = 10        # GAD-7 score at/above this = concern

    # Self-Care Score (Phase 22)
    SELFCARE_SCORE_ENABLED: bool = True                # master toggle
    SELFCARE_MOOD_WINDOW_DAYS: int = 14                # days of mood history to consider
    SELFCARE_STREAK_CAP_DAYS: int = 14                 # max streak days for full score
    SELFCARE_BURNOUT_ENGAGEMENT_THRESHOLD: float = 80.0   # engagement above this + low effectiveness = burnout
    SELFCARE_BURNOUT_EFFECTIVENESS_THRESHOLD: float = 30.0  # effectiveness below this + high engagement = burnout


def _load_config() -> AppConfig:
    """Load configuration from environment variables."""

    def _env_int(key: str, default: int) -> int:
        val = os.getenv(key)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            logger.warning(f"Invalid integer for {key}={val}, using default {default}")
            return default

    def _env_float(key: str, default: float) -> float:
        val = os.getenv(key)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            logger.warning(f"Invalid float for {key}={val}, using default {default}")
            return default

    def _env_bool(key: str, default: bool) -> bool:
        val = os.getenv(key)
        if val is None:
            return default
        return val.lower() in ("true", "1", "yes")

    return AppConfig(
        ENVIRONMENT=os.getenv("ENVIRONMENT", "development"),
        OPENAI_MODEL=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        FAISS_INDEX_PATH=os.getenv("FAISS_INDEX_PATH", "./faiss_vecdb"),
        RATE_LIMIT_CHAT=_env_int("RATE_LIMIT_CHAT", 10),
        RATE_LIMIT_GLOBAL=_env_int("RATE_LIMIT_GLOBAL", 100),
        RATE_LIMIT_WINDOW_SECONDS=_env_int("RATE_LIMIT_WINDOW_SECONDS", 60),
        CACHE_TTL_USER_PROFILE=_env_int("CACHE_TTL_USER_PROFILE", 60),
        CACHE_TTL_EFFECTIVENESS=_env_int("CACHE_TTL_EFFECTIVENESS", 120),
        CACHE_TTL_EMOTION=_env_int("CACHE_TTL_EMOTION", 300),
        CACHE_MAX_SIZE=_env_int("CACHE_MAX_SIZE", 500),
        LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
        LOG_FORMAT=os.getenv("LOG_FORMAT", "json"),
        METRICS_ENABLED=_env_bool("METRICS_ENABLED", True),
        # Dependency Detection
        DEPENDENCY_MAX_MESSAGES_PER_HOUR=_env_int("DEPENDENCY_MAX_MESSAGES_PER_HOUR", 20),
        DEPENDENCY_MAX_SESSIONS_PER_DAY=_env_int("DEPENDENCY_MAX_SESSIONS_PER_DAY", 8),
        DEPENDENCY_NIGHTTIME_START_HOUR=_env_int("DEPENDENCY_NIGHTTIME_START_HOUR", 0),
        DEPENDENCY_NIGHTTIME_END_HOUR=_env_int("DEPENDENCY_NIGHTTIME_END_HOUR", 5),
        DEPENDENCY_NIGHTTIME_THRESHOLD=_env_int("DEPENDENCY_NIGHTTIME_THRESHOLD", 5),
        DEPENDENCY_ESCALATION_RATIO=_env_float("DEPENDENCY_ESCALATION_RATIO", 1.5),
        DEPENDENCY_CONSECUTIVE_DAYS_THRESHOLD=_env_int("DEPENDENCY_CONSECUTIVE_DAYS_THRESHOLD", 14),
        DEPENDENCY_COOLDOWN_MINUTES=_env_int("DEPENDENCY_COOLDOWN_MINUTES", 30),
        # Cultural Competence
        CULTURAL_BIAS_CHECK_ENABLED=_env_bool("CULTURAL_BIAS_CHECK_ENABLED", True),
        CULTURAL_DEFAULT_COUNTRY=os.getenv("CULTURAL_DEFAULT_COUNTRY", "US"),
        # Data Retention
        RETENTION_CHAT_SESSIONS_DAYS=_env_int("RETENTION_CHAT_SESSIONS_DAYS", 365),
        RETENTION_MEMORIES_DAYS=_env_int("RETENTION_MEMORIES_DAYS", 730),
        RETENTION_FEEDBACK_DAYS=_env_int("RETENTION_FEEDBACK_DAYS", 365),
        RETENTION_EXERCISE_SESSIONS_DAYS=_env_int("RETENTION_EXERCISE_SESSIONS_DAYS", 365),
        RETENTION_SOUNDSCAPE_SESSIONS_DAYS=_env_int("RETENTION_SOUNDSCAPE_SESSIONS_DAYS", 180),
        RETENTION_SAFETY_EVENTS_DAYS=_env_int("RETENTION_SAFETY_EVENTS_DAYS", 730),
        RETENTION_INTERACTION_METRICS_DAYS=_env_int("RETENTION_INTERACTION_METRICS_DAYS", 365),
        RETENTION_AUDIT_LOG_DAYS=_env_int("RETENTION_AUDIT_LOG_DAYS", 2555),
        # Audit
        AUDIT_LOG_ENABLED=_env_bool("AUDIT_LOG_ENABLED", True),
        # Audio Storage (GCS)
        GCS_AUDIO_BUCKET=os.getenv("GCS_AUDIO_BUCKET", ""),
        GCS_AUDIO_PREFIX=os.getenv("GCS_AUDIO_PREFIX", "soundscapes/"),
        GCS_SIGNED_URL_EXPIRY_MINUTES=_env_int("GCS_SIGNED_URL_EXPIRY_MINUTES", 60),
        LOCAL_AUDIO_FALLBACK=_env_bool("LOCAL_AUDIO_FALLBACK", False),
        LOCAL_AUDIO_DIR=os.getenv("LOCAL_AUDIO_DIR", ""),
        # Voice I/O
        TTS_PROVIDER=os.getenv("TTS_PROVIDER", "edge-tts"),
        STT_PROVIDER=os.getenv("STT_PROVIDER", "speech_recognition"),
        TTS_VOICE=os.getenv("TTS_VOICE", "en-US-AriaNeural"),
        TTS_RATE=os.getenv("TTS_RATE", "+0%"),
        MAX_AUDIO_SIZE_MB=_env_int("MAX_AUDIO_SIZE_MB", 10),
        # Wearable Integration
        WEARABLE_SYNC_ENABLED=_env_bool("WEARABLE_SYNC_ENABLED", True),
        WEARABLE_HEALTH_CONTEXT_DAYS=_env_int("WEARABLE_HEALTH_CONTEXT_DAYS", 7),
        RETENTION_HEALTH_METRICS_DAYS=_env_int("RETENTION_HEALTH_METRICS_DAYS", 365),
        # RL / Thompson Sampling
        RL_ENABLED=_env_bool("RL_ENABLED", True),
        RL_EXPLORATION_BONUS=_env_float("RL_EXPLORATION_BONUS", 0.5),
        RL_MIN_OUTCOMES_FOR_RL=_env_int("RL_MIN_OUTCOMES_FOR_RL", 3),
        RL_SUCCESS_THRESHOLD=_env_int("RL_SUCCESS_THRESHOLD", 4),
        RL_CACHE_TTL=_env_int("RL_CACHE_TTL", 300),
        # Fine-Tuning (Phase 18)
        FT_ENABLED=_env_bool("FT_ENABLED", False),
        FT_MODEL_ID=os.getenv("FT_MODEL_ID", ""),
        FT_BASE_MODEL=os.getenv("FT_BASE_MODEL", "gpt-4o-mini"),
        FT_AB_SPLIT_PERCENT=_env_int("FT_AB_SPLIT_PERCENT", 50),
        FT_MIN_TRAINING_EXAMPLES=_env_int("FT_MIN_TRAINING_EXAMPLES", 100),
        FT_MIN_FEEDBACK_RATING=os.getenv("FT_MIN_FEEDBACK_RATING", "helpful"),
        FT_MIN_HELPFULNESS_SCORE=_env_int("FT_MIN_HELPFULNESS_SCORE", 4),
        FT_TRAINING_EPOCHS=_env_int("FT_TRAINING_EPOCHS", 3),
        FT_RETENTION_TRAINING_DATA_DAYS=_env_int("FT_RETENTION_TRAINING_DATA_DAYS", 365),
        # Monitoring Dashboard (Phase 19)
        DASHBOARD_ENABLED=_env_bool("DASHBOARD_ENABLED", True),
        DASHBOARD_CACHE_TTL=_env_int("DASHBOARD_CACHE_TTL", 60),
        DASHBOARD_REFRESH_INTERVAL=_env_int("DASHBOARD_REFRESH_INTERVAL", 30),
        # Human Escalation & Reviews (Phase 20)
        ESCALATION_ENABLED=_env_bool("ESCALATION_ENABLED", True),
        ESCALATION_REPEATED_HIGH_THRESHOLD=_env_int("ESCALATION_REPEATED_HIGH_THRESHOLD", 3),
        ESCALATION_REPEATED_HIGH_WINDOW_DAYS=_env_int("ESCALATION_REPEATED_HIGH_WINDOW_DAYS", 7),
        REVIEW_ENABLED=_env_bool("REVIEW_ENABLED", True),
        REVIEW_DEFAULT_PERIOD_DAYS=_env_int("REVIEW_DEFAULT_PERIOD_DAYS", 365),
        # Assessment System (Phase 21)
        ASSESSMENT_ENABLED=_env_bool("ASSESSMENT_ENABLED", True),
        ASSESSMENT_REMINDER_DAYS=_env_int("ASSESSMENT_REMINDER_DAYS", 14),
        ASSESSMENT_WHO5_CONCERN_THRESHOLD=_env_int("ASSESSMENT_WHO5_CONCERN_THRESHOLD", 50),
        ASSESSMENT_PHQ9_CONCERN_THRESHOLD=_env_int("ASSESSMENT_PHQ9_CONCERN_THRESHOLD", 10),
        ASSESSMENT_GAD7_CONCERN_THRESHOLD=_env_int("ASSESSMENT_GAD7_CONCERN_THRESHOLD", 10),
        # Self-Care Score (Phase 22)
        SELFCARE_SCORE_ENABLED=_env_bool("SELFCARE_SCORE_ENABLED", True),
        SELFCARE_MOOD_WINDOW_DAYS=_env_int("SELFCARE_MOOD_WINDOW_DAYS", 14),
        SELFCARE_STREAK_CAP_DAYS=_env_int("SELFCARE_STREAK_CAP_DAYS", 14),
        SELFCARE_BURNOUT_ENGAGEMENT_THRESHOLD=_env_float("SELFCARE_BURNOUT_ENGAGEMENT_THRESHOLD", 80.0),
        SELFCARE_BURNOUT_EFFECTIVENESS_THRESHOLD=_env_float("SELFCARE_BURNOUT_EFFECTIVENESS_THRESHOLD", 30.0),
    )


# ── Singleton ─────────────────────────────────────────────

_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create AppConfig singleton."""
    global _config
    if _config is None:
        _config = _load_config()
    return _config
