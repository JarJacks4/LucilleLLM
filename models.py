"""
LucilleLLM - Data Models

Pydantic models for user profiles (5 behavioral layers), chat request/response,
onboarding, and API schemas. Used across main.py, user_service.py, and firebase_service.py.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


# ── Enums ──────────────────────────────────────────────

class CommunicationStyle(str, Enum):
    DIRECT = "direct"
    EMPATHETIC = "empathetic"
    ANALYTICAL = "analytical"
    CASUAL = "casual"


class DetectedEmotion(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"
    HOPELESS = "hopeless"
    LONELY = "lonely"
    OVERWHELMED = "overwhelmed"
    GRATEFUL = "grateful"


class UserIntent(str, Enum):
    VENTING = "venting"
    SEEKING_ADVICE = "seeking_advice"
    CRISIS = "crisis"
    CASUAL_CHAT = "casual_chat"
    DOING_EXERCISE = "doing_exercise"
    REFLECTING = "reflecting"


# ── Layer 1: Cognitive Layer ───────────────────────────

class CognitiveLayer(BaseModel):
    """Captures user beliefs and thought patterns"""
    beliefs: List[str] = Field(default_factory=list,
        description="Core beliefs, e.g. 'I believe in work-life balance'")
    thought_patterns: List[str] = Field(default_factory=list,
        description="Recurring thought patterns, e.g. 'catastrophizing', 'all-or-nothing'")
    cognitive_distortions: List[str] = Field(default_factory=list,
        description="Identified cognitive distortions")
    worldview_notes: str = Field(default="",
        description="Freeform notes about the user's worldview")


# ── Layer 2: Affective (Emotional) Layer ──────────────

class MoodEntry(BaseModel):
    """A single mood observation"""
    mood: str  # e.g. "anxious", "happy", "neutral"
    intensity: int = Field(default=5, ge=1, le=10,
        description="1=barely noticeable, 10=overwhelming")
    context: str = Field(default="",
        description="What triggered or accompanied this mood")
    recorded_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    # Phase 2: detection metadata (backward-compatible defaults)
    detected_via: str = Field(default="manual",
        description="'text_auto', 'image_auto', or 'manual'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0,
        description="Detection confidence, 1.0 for manual entries")
    intent: Optional[str] = Field(default=None,
        description="Detected user intent, e.g. 'venting', 'seeking_advice'")


class AffectiveLayer(BaseModel):
    """Tracks current mood, emotions, emotional cues over time"""
    current_mood: str = Field(default="neutral")
    mood_history: List[MoodEntry] = Field(default_factory=list,
        description="Chronological mood entries, capped at 50")
    emotional_triggers: List[str] = Field(default_factory=list,
        description="Known emotional triggers, e.g. 'deadlines', 'social situations'")


# ── Layer 3: Behavioral Layer ─────────────────────────

class HabitEntry(BaseModel):
    """A tracked habit or routine"""
    name: str  # e.g. "morning_exercise", "meditation"
    frequency: str = Field(default="daily",
        description="daily, weekly, occasionally, rarely")
    status: str = Field(default="active",
        description="active, paused, dropped")
    notes: str = ""


class BehavioralLayer(BaseModel):
    """Represents habits, tasks, and actions"""
    habits: List[HabitEntry] = Field(default_factory=list)
    sleep_pattern: str = Field(default="",
        description="e.g. '11pm-7am', 'irregular'")
    exercise_frequency: str = Field(default="",
        description="e.g. 'daily', '3x/week', 'rarely'")
    noted_behaviors: List[str] = Field(default_factory=list,
        description="Other behavioral observations")


# ── Layer 4: Motivational / Goal Layer ────────────────

class Goal(BaseModel):
    """A specific user goal"""
    goal_id: str = Field(default="")
    title: str
    description: str = ""
    category: str = Field(default="general",
        description="e.g. 'health', 'career', 'relationships', 'self-care'")
    status: str = Field(default="active",
        description="active, completed, abandoned")
    progress_notes: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class MotivationalLayer(BaseModel):
    """Encodes user motivations, values, and goals"""
    core_values: List[str] = Field(default_factory=list,
        description="e.g. 'health', 'family', 'creativity'")
    motivations: List[str] = Field(default_factory=list,
        description="What drives the user")
    goals: List[Goal] = Field(default_factory=list)


# ── Layer 5: Persona / Profile Layer ──────────────────

class PersonaLayer(BaseModel):
    """Stable traits, communication preferences, background"""
    display_name: str = Field(default="",
        description="How the user wants to be addressed")
    personality_traits: List[str] = Field(default_factory=list,
        description="e.g. 'introvert', 'detail-oriented'")
    communication_preference: CommunicationStyle = Field(
        default=CommunicationStyle.EMPATHETIC)
    cultural_background: str = Field(default="",
        description="Optional; helps tailor advice")
    age_range: str = Field(default="",
        description="e.g. '25-34', 'prefer not to say'")
    interests: List[str] = Field(default_factory=list)


# ── Composite User Profile ───────────────────────────

class UserProfile(BaseModel):
    """Complete user profile combining all 5 behavioral layers"""
    user_id: str
    cognitive: CognitiveLayer = Field(default_factory=CognitiveLayer)
    affective: AffectiveLayer = Field(default_factory=AffectiveLayer)
    behavioral: BehavioralLayer = Field(default_factory=BehavioralLayer)
    motivational: MotivationalLayer = Field(default_factory=MotivationalLayer)
    persona: PersonaLayer = Field(default_factory=PersonaLayer)
    onboarding_completed: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── API Request / Response Models ─────────────────────

class OnboardingRequest(BaseModel):
    """Collected during the onboarding flow"""
    user_id: Optional[str] = None  # auto-generated if absent
    display_name: str = ""
    age_range: str = ""
    personality_traits: List[str] = Field(default_factory=list)
    communication_preference: CommunicationStyle = CommunicationStyle.EMPATHETIC
    interests: List[str] = Field(default_factory=list)
    core_values: List[str] = Field(default_factory=list)
    current_mood: str = "neutral"
    goals: List[str] = Field(default_factory=list,
        description="Simple goal titles to bootstrap the goal layer")
    sleep_pattern: str = ""
    exercise_frequency: str = ""


class OnboardingResponse(BaseModel):
    user_id: str
    status: str = "success"
    message: str = ""
    timestamp: str = ""


class UserProfileResponse(BaseModel):
    user_id: str
    profile: UserProfile
    status: str = "success"
    timestamp: str = ""


class UserProfileUpdateRequest(BaseModel):
    """Partial update; only non-None fields are applied"""
    cognitive: Optional[CognitiveLayer] = None
    affective: Optional[AffectiveLayer] = None
    behavioral: Optional[BehavioralLayer] = None
    motivational: Optional[MotivationalLayer] = None
    persona: Optional[PersonaLayer] = None


# ── Memory System ─────────────────────────────────────

class MemoryType(str, Enum):
    EPISODIC = "episodic"      # Specific events/experiences
    SEMANTIC = "semantic"      # General knowledge about the user
    FACT = "fact"              # Structured facts (name, location, etc.)


class Memory(BaseModel):
    """A single memory entry stored per user"""
    memory_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    content: str = Field(description="The memory text")
    importance: int = Field(default=5, ge=1, le=10,
        description="1=trivial, 10=critical life event")
    tags: List[str] = Field(default_factory=list,
        description="e.g. 'family', 'work', 'health'")
    source: str = Field(default="auto_extracted",
        description="'auto_extracted', 'manual', 'onboarding'")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = Field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = Field(default=0)


class MemoryCreateRequest(BaseModel):
    """Manual memory creation"""
    content: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    importance: int = Field(default=5, ge=1, le=10)
    tags: List[str] = Field(default_factory=list)


class MemorySearchRequest(BaseModel):
    """Search query for memories"""
    query: str
    limit: int = Field(default=5, ge=1, le=20)


# ── Therapy Module System ─────────────────────────────

class TherapyModality(str, Enum):
    CBT = "cbt"        # Cognitive Behavioral Therapy
    ACT = "act"        # Acceptance and Commitment Therapy
    DBT = "dbt"        # Dialectical Behavior Therapy
    MI = "mi"          # Motivational Interviewing


class ExerciseTemplate(BaseModel):
    """A structured therapy exercise template"""
    exercise_id: str
    modality: TherapyModality
    title: str
    description: str
    target_emotions: List[str] = Field(default_factory=list,
        description="Emotions this exercise helps with, e.g. ['anxious', 'overwhelmed']")
    target_intents: List[str] = Field(default_factory=list,
        description="Intents this exercise suits, e.g. ['seeking_advice', 'doing_exercise']")
    difficulty: str = Field(default="beginner",
        description="beginner, intermediate, advanced")
    duration_minutes: int = Field(default=10)
    steps: List[str] = Field(default_factory=list,
        description="Ordered steps to guide the user through")
    system_prompt_addon: str = Field(default="",
        description="Instructions injected into the system prompt when active")
    # Phase 6: Practice task auto-generation
    practice_prompt: str = Field(default="",
        description="Template for auto-generated practice task when exercise completes. "
                    "Empty means no auto-task.")
    practice_target_count: int = Field(default=3,
        description="Default target count for the auto-generated practice task")
    practice_days: int = Field(default=3,
        description="Number of days after completion to set as due date")


class ExerciseSession(BaseModel):
    """Tracks an active exercise session for a user"""
    session_id: str
    user_id: str
    exercise_id: str
    modality: str
    title: str
    current_step: int = Field(default=0)
    total_steps: int = Field(default=0)
    status: str = Field(default="active",
        description="active, completed, abandoned")
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    notes: List[str] = Field(default_factory=list,
        description="User responses or notes collected during the exercise")


class ExerciseRecommendation(BaseModel):
    """A recommended exercise with relevance context"""
    exercise_id: str
    modality: str
    title: str
    description: str
    reason: str = Field(description="Why this exercise is recommended")
    difficulty: str = "beginner"
    duration_minutes: int = 10
    # Phase 17: RL metadata (optional, backward-compatible)
    rl_meta: Optional["RLRecommendationMeta"] = Field(default=None,
        description="Thompson Sampling metadata, if RL was used")


class StartExerciseRequest(BaseModel):
    """Request to start a therapy exercise"""
    exercise_id: str


class StartExerciseResponse(BaseModel):
    """Response when starting an exercise"""
    session_id: str
    exercise_id: str
    modality: str
    title: str
    total_steps: int
    first_step: str
    status: str = "success"
    timestamp: str = ""
    # Phase 8: Suggested soundscape for this exercise
    suggested_soundscape: Optional[dict] = Field(default=None,
        description="Suggested soundscape to play during this exercise")


# ── Task / Homework System ───────────────────────────

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    EXPIRED = "expired"


class PracticeTask(BaseModel):
    """A between-session homework/practice assignment."""
    task_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str
    source_exercise_id: str
    source_session_id: str = Field(default="",
        description="The exercise session that triggered this task")
    modality: str
    title: str = Field(description="Short task title, e.g. 'Practice: Thought Record'")
    description: str = Field(default="",
        description="Detailed instructions for the practice")
    status: TaskStatus = TaskStatus.PENDING
    assigned_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    due_date: str = Field(default="",
        description="ISO date string for when this task is due")
    completed_at: Optional[str] = None
    target_count: int = Field(default=1,
        description="How many times to practice")
    completed_count: int = Field(default=0,
        description="How many times the user has actually practiced")
    notes: List[str] = Field(default_factory=list,
        description="User reflections or notes per practice attempt")


class CreateTaskRequest(BaseModel):
    """Manual task creation by the API/frontend."""
    source_exercise_id: str
    title: str
    description: str = ""
    due_date: str = ""
    target_count: int = 1


class UpdateTaskRequest(BaseModel):
    """Update task status or add notes."""
    status: Optional[TaskStatus] = None
    completed_count: Optional[int] = None
    note: Optional[str] = None


# ── Progress Analytics ───────────────────────────────

class ProgressSummary(BaseModel):
    """Computed analytics from exercise and task history."""
    user_id: str
    total_exercises_started: int = 0
    total_exercises_completed: int = 0
    total_exercises_abandoned: int = 0
    completion_rate: float = Field(default=0.0,
        description="Fraction of started exercises that were completed")
    modality_counts: dict = Field(default_factory=dict,
        description="Completed exercises per modality, e.g. {'cbt': 5, 'dbt': 2}")
    current_streak_days: int = Field(default=0,
        description="Consecutive days with at least one completed exercise or task")
    longest_streak_days: int = 0
    total_tasks_assigned: int = 0
    total_tasks_completed: int = 0
    task_completion_rate: float = 0.0
    total_practice_minutes: int = Field(default=0,
        description="Estimated total minutes from completed exercise durations")
    sessions_this_week: int = Field(default=0,
        description="Number of exercises completed in the current week (last 7 days)")
    computed_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    # Phase 7: Feedback stats
    total_feedback_given: int = Field(default=0,
        description="Total response feedback entries submitted")
    average_helpfulness: float = Field(default=0.0,
        description="Average helpfulness score from exercise outcomes")
    favorite_modality: str = Field(default="",
        description="Modality with highest effectiveness score")


# ── Feedback & Closed-Loop System ──────────────────────

class FeedbackRating(str, Enum):
    HELPFUL = "helpful"
    NEUTRAL = "neutral"
    NOT_HELPFUL = "not_helpful"


class ResponseFeedback(BaseModel):
    """Rating for an individual chat response."""
    feedback_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str
    session_id: str
    message_index: int = Field(default=0,
        description="Which assistant message in the session (0 = latest)")
    rating: FeedbackRating
    comment: str = Field(default="",
        description="Optional freeform comment on why they rated this way")
    detected_emotion: str = Field(default="",
        description="User's emotion at time of feedback")
    detected_intent: str = Field(default="",
        description="User's intent at time of feedback")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExerciseOutcome(BaseModel):
    """Post-exercise outcome data submitted by the user."""
    outcome_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str
    session_id: str
    exercise_id: str
    modality: str
    mood_before: int = Field(default=5, ge=1, le=10,
        description="Mood before the exercise, 1=terrible, 10=excellent")
    mood_after: int = Field(default=5, ge=1, le=10,
        description="Mood after the exercise, 1=terrible, 10=excellent")
    helpfulness: int = Field(default=3, ge=1, le=5,
        description="How helpful was this exercise, 1=not at all, 5=extremely")
    would_repeat: bool = Field(default=True,
        description="Whether the user would do this exercise again")
    comment: str = Field(default="",
        description="Optional reflection on the exercise")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class SubmitFeedbackRequest(BaseModel):
    """Request body for submitting response feedback."""
    session_id: str
    message_index: int = Field(default=0)
    rating: FeedbackRating
    comment: str = Field(default="")


class SubmitOutcomeRequest(BaseModel):
    """Request body for submitting exercise outcome."""
    session_id: str
    mood_before: int = Field(ge=1, le=10)
    mood_after: int = Field(ge=1, le=10)
    helpfulness: int = Field(ge=1, le=5)
    would_repeat: bool = Field(default=True)
    comment: str = Field(default="")


class EffectivenessProfile(BaseModel):
    """Computed effectiveness scores from outcome data."""
    user_id: str
    modality_scores: dict = Field(default_factory=dict,
        description="Average helpfulness per modality, e.g. {'cbt': 4.2, 'dbt': 3.8}")
    exercise_scores: dict = Field(default_factory=dict,
        description="Average helpfulness per exercise, e.g. {'cbt_thought_record': 4.5}")
    modality_mood_deltas: dict = Field(default_factory=dict,
        description="Average mood improvement per modality, e.g. {'cbt': 2.1}")
    favorite_modality: str = Field(default="",
        description="Modality with highest average helpfulness")
    total_outcomes: int = Field(default=0)
    total_response_feedback: int = Field(default=0)
    response_helpful_rate: float = Field(default=0.0,
        description="Fraction of response feedback rated 'helpful'")
    computed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── RL / Thompson Sampling (Phase 17) ────────────────


class BanditArmState(BaseModel):
    """State of a single Thompson Sampling arm."""
    exercise_id: str
    emotion_group: str
    alpha: float = Field(default=1.0, description="Success count + prior")
    beta: float = Field(default=1.0, description="Failure count + prior")
    mean_success_rate: float = Field(default=0.5,
        description="alpha / (alpha + beta)")
    total_observations: float = Field(default=0.0,
        description="Total reward updates (alpha + beta - 2 for prior)")


class RLRecommendationMeta(BaseModel):
    """Metadata about the RL selection for a recommendation."""
    rl_used: bool = Field(default=False,
        description="Whether Thompson Sampling was used")
    emotion_group: str = Field(default="",
        description="Emotion group used for bandit context")
    thompson_score: float = Field(default=0.0,
        description="Sampled Thompson score (0-1)")
    arm_alpha: float = 1.0
    arm_beta: float = 1.0


# ── Soundscape & Audio Engine ────────────────────────

class SoundscapeCategory(str, Enum):
    NATURE = "nature"
    AMBIENT = "ambient"
    MEDITATION = "meditation"
    BINAURAL = "binaural"
    MUSIC = "music"


class SoundscapeTemplate(BaseModel):
    """A soundscape audio template for relaxation/focus."""
    soundscape_id: str
    category: SoundscapeCategory
    title: str
    description: str
    duration_seconds: int = Field(default=300,
        description="Loop duration in seconds (default 5 minutes)")
    target_emotions: List[str] = Field(default_factory=list,
        description="Emotions this soundscape helps with, e.g. ['anxious', 'overwhelmed']")
    target_contexts: List[str] = Field(default_factory=list,
        description="Usage contexts, e.g. ['relaxation', 'focus', 'sleep']")
    audio_url: str = Field(default="",
        description="Placeholder URL for frontend audio asset")
    icon: str = Field(default="",
        description="Emoji icon for UI display")


class SoundscapeSession(BaseModel):
    """Tracks a user's soundscape listening session."""
    session_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str
    soundscape_id: str
    title: str
    category: str
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    stopped_at: Optional[str] = Field(default=None,
        description="None means session is still active")
    duration_seconds: int = Field(default=0,
        description="Computed when session is stopped")


class SoundscapeRecommendation(BaseModel):
    """A recommended soundscape with relevance context."""
    soundscape_id: str
    category: str
    title: str
    description: str
    reason: str = Field(description="Why this soundscape is recommended")
    audio_url: str = ""
    icon: str = ""


class StartSoundscapeRequest(BaseModel):
    """Request to start a soundscape session."""
    soundscape_id: str


# ── Safety & Ethics System ───────────────────────────

class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyEventType(str, Enum):
    CRISIS_DETECTED = "crisis_detected"
    HIGH_RISK_INPUT = "high_risk_input"
    OUTPUT_MODIFIED = "output_modified"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    CRITICAL_INTERCEPT = "critical_intercept"


class CrisisResourceType(str, Enum):
    HOTLINE = "hotline"
    TEXT = "text"
    CHAT = "chat"
    WEBSITE = "website"


class CrisisResource(BaseModel):
    """A crisis helpline or resource."""
    name: str
    number: str = Field(default="",
        description="Phone number or text code")
    description: str = ""
    resource_type: CrisisResourceType = CrisisResourceType.HOTLINE
    url: str = ""
    country: str = Field(default="US")


class SafetyCheckResult(BaseModel):
    """Result of a safety screening on input or output text."""
    risk_level: RiskLevel = RiskLevel.LOW
    flags: List[str] = Field(default_factory=list,
        description="Triggered flag reasons, e.g. 'self_harm_keyword', 'jailbreak_pattern'")
    crisis_detected: bool = False
    jailbreak_detected: bool = False
    helplines_needed: bool = False
    action_taken: str = Field(default="none",
        description="none, prompt_enhanced, response_modified, crisis_intercept")


class SafetyEvent(BaseModel):
    """Audit log entry for a safety-related event."""
    event_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str
    session_id: str = ""
    event_type: SafetyEventType
    risk_level: RiskLevel
    message_snippet: str = Field(default="",
        description="First 200 chars of the triggering message")
    flags: List[str] = Field(default_factory=list)
    action_taken: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class SafetyCheckRequest(BaseModel):
    """Request body for manual /safety/check endpoint."""
    text: str
    check_type: str = Field(default="input",
        description="'input' for user message screening, 'output' for bot response screening")


# ── Dependency & Anti-Dependency System ──────────────

class DependencyRiskLevel(str, Enum):
    """Risk level for app dependency assessment."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class DependencySignal(str, Enum):
    """Specific signals indicating potential dependency on the app."""
    ESCALATING_FREQUENCY = "escalating_frequency"
    EXCESSIVE_SESSION_LENGTH = "excessive_session_length"
    NIGHTTIME_USAGE = "nighttime_usage"
    VALIDATION_SEEKING = "validation_seeking"
    MINOR_ISSUE_CRISIS = "minor_issue_crisis"
    CONSECUTIVE_DAYS_EXCESSIVE = "consecutive_days_excessive"
    HIGH_MESSAGES_PER_HOUR = "high_messages_per_hour"
    HIGH_SESSIONS_PER_DAY = "high_sessions_per_day"


class InteractionMetrics(BaseModel):
    """Lightweight counters for a user's interaction frequency.
    Stored as a single Firestore document per user for fast reads/writes."""
    user_id: str
    messages_today: int = 0
    sessions_today: int = 0
    messages_this_hour: int = 0
    consecutive_days: int = 0
    total_messages_this_week: int = 0
    total_messages_last_week: int = 0
    nighttime_messages_count: int = Field(default=0,
        description="Messages sent between midnight-5am in current week")
    last_message_at: str = Field(default="",
        description="ISO timestamp of last message")
    last_reset_date: str = Field(default="",
        description="ISO date string of last daily counter reset")
    last_hour_reset: str = Field(default="",
        description="ISO hour string of last hourly counter reset")
    week_start_date: str = Field(default="",
        description="ISO date string of current week start for weekly tracking")
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class DependencyAssessment(BaseModel):
    """Result of a dependency risk evaluation."""
    user_id: str
    risk_level: DependencyRiskLevel = DependencyRiskLevel.NONE
    signals: List[str] = Field(default_factory=list,
        description="Detected dependency signals")
    score: int = Field(default=0, ge=0, le=100,
        description="Composite dependency risk score 0-100")
    boundary_message: str = Field(default="",
        description="Compassionate boundary-setting message if needed")
    cooldown_suggested: bool = False
    assessed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Cultural Competence System ───────────────────────

class CulturalContext(BaseModel):
    """Cultural context information derived from user profile."""
    user_id: str = ""
    cultural_background: str = Field(default="",
        description="From user's persona layer")
    country_code: str = Field(default="US",
        description="Detected or specified country for crisis resource selection")
    language_preference: str = Field(default="en")
    bias_flags: List[str] = Field(default_factory=list,
        description="Detected bias patterns in output text")
    culturally_appropriate: bool = True


# ── GDPR/HIPAA Compliance System ─────────────────────

class ConsentCategory(str, Enum):
    """Categories of data processing requiring user consent."""
    DATA_COLLECTION = "data_collection"
    AI_PROCESSING = "ai_processing"
    MEMORY_STORAGE = "memory_storage"
    EMOTION_DETECTION = "emotion_detection"
    ANALYTICS = "analytics"
    CRISIS_DETECTION = "crisis_detection"
    ML_TRAINING = "ml_training"              # Phase 18: consent for fine-tuning data usage


class ConsentRecord(BaseModel):
    """Tracks user consent for each data processing category."""
    user_id: str
    consents: dict = Field(default_factory=dict,
        description="Map of ConsentCategory -> bool")
    privacy_policy_version: str = Field(default="1.0",
        description="Version of privacy policy the user agreed to")
    consented_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    ip_address: str = Field(default="",
        description="IP address at time of consent for legal record")


class ConsentRequest(BaseModel):
    """Request body for recording or updating consent."""
    consents: dict = Field(
        description="Map of category -> bool, e.g. {'data_collection': true}")
    privacy_policy_version: str = Field(default="1.0")


class ConsentResponse(BaseModel):
    """Response for consent operations."""
    user_id: str
    consents: dict
    privacy_policy_version: str
    status: str = "success"
    timestamp: str = ""


class AuditAction(str, Enum):
    """Types of data access actions logged for compliance."""
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    CONSENT_CHANGE = "consent_change"


class AuditLogEntry(BaseModel):
    """Immutable audit log entry. NOT keyed to user_id (survives deletion)."""
    log_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str = Field(description="The user whose data was accessed")
    actor_id: str = Field(default="system",
        description="Who performed the action (user_id, 'system', or 'admin')")
    action: AuditAction
    resource_type: str = Field(
        description="e.g. 'user_profile', 'chat_session', 'memory', 'consent'")
    resource_id: str = Field(default="",
        description="Specific document ID if applicable")
    details: str = Field(default="",
        description="Additional context about the operation")
    ip_address: str = Field(default="")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class DataExportResponse(BaseModel):
    """Container for full user data export (GDPR Art. 20)."""
    user_id: str
    export_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    profile: Optional[dict] = None
    memories: List[dict] = Field(default_factory=list)
    chat_sessions: List[dict] = Field(default_factory=list)
    response_feedback: List[dict] = Field(default_factory=list)
    exercise_outcomes: List[dict] = Field(default_factory=list)
    exercise_sessions: List[dict] = Field(default_factory=list)
    practice_tasks: List[dict] = Field(default_factory=list)
    soundscape_sessions: List[dict] = Field(default_factory=list)
    safety_events: List[dict] = Field(default_factory=list)
    health_metrics: List[dict] = Field(default_factory=list)
    bandit_state: Optional[dict] = None
    training_examples: List[dict] = Field(default_factory=list)
    model_performance: List[dict] = Field(default_factory=list)
    interaction_metrics: Optional[dict] = None
    consent: Optional[dict] = None
    escalation_events: List[dict] = Field(default_factory=list)
    annual_reviews: List[dict] = Field(default_factory=list)
    status: str = "success"


class DeletionReceipt(BaseModel):
    """Receipt documenting what was deleted during cascade deletion (GDPR Art. 17)."""
    user_id: str
    deletion_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    collections_deleted: dict = Field(default_factory=dict,
        description="Map of collection name -> count of docs deleted")
    total_documents_deleted: int = 0
    cache_cleared: bool = False
    status: str = "success"


# ── Emotion Detection Result (internal) ──────────────

class EmotionDetectionResult(BaseModel):
    """Internal result from text-based emotion/intent detection"""
    emotion: str = "neutral"
    intensity: int = 5
    intent: str = "casual_chat"
    confidence: float = 0.0


# ── Chat Models (updated, backward-compatible) ────────

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)
    session_id: str = Field(..., max_length=200)
    user_id: Optional[str] = Field(default=None, max_length=100)

class ChatResponse(BaseModel):
    session_id: str
    response: str
    conversation: List[str]
    status: str = "success"
    timestamp: str = ""
    message_count: int = 0
    user_id: Optional[str] = None
    detected_emotion: Optional[str] = None
    detected_intent: Optional[str] = None
    model_used: Optional[str] = None         # Phase 18: which model generated this response


# ── Voice Chat Models ────────────────────────────────────


class VoiceChatRequest(BaseModel):
    """Chat request that accepts optional audio input.

    If audio_input is provided, it is transcribed to text and used as the message.
    If both message and audio_input are provided, audio_input takes precedence.
    """
    message: str = Field(default="",
        description="Text message (used if audio_input is absent)")
    session_id: str
    user_id: Optional[str] = None
    audio_input: Optional[str] = Field(default=None,
        description="Base64-encoded audio data")
    audio_format: str = Field(default="wav",
        description="Audio format: wav, flac, mp3, ogg, webm")
    response_format: str = Field(default="both",
        description="Response format: 'text', 'audio', or 'both'")
    tts_voice: Optional[str] = Field(default=None,
        description="Override TTS voice (e.g. 'en-US-GuyNeural')")


class VoiceChatResponse(BaseModel):
    """Chat response with optional audio output."""
    session_id: str
    response: str
    conversation: List[str]
    status: str = "success"
    timestamp: str = ""
    message_count: int = 0
    user_id: Optional[str] = None
    detected_emotion: Optional[str] = None
    detected_intent: Optional[str] = None
    # Voice-specific fields
    transcribed_text: Optional[str] = Field(default=None,
        description="Text transcribed from audio input (None if text input)")
    audio_output: Optional[str] = Field(default=None,
        description="Base64-encoded MP3 audio of the response")
    audio_duration_ms: Optional[int] = Field(default=None,
        description="Estimated duration of audio output in milliseconds")


class TTSRequest(BaseModel):
    """Simple text-to-speech request."""
    text: str = Field(description="Text to convert to speech")
    voice: Optional[str] = Field(default=None,
        description="Voice name (e.g. 'en-US-AriaNeural')")
    rate: Optional[str] = Field(default=None,
        description="Speech rate (e.g. '+10%', '-5%')")


class TTSResponse(BaseModel):
    """Text-to-speech response."""
    audio: str = Field(description="Base64-encoded MP3 audio")
    duration_ms: int = Field(default=0,
        description="Estimated audio duration in milliseconds")
    voice: str = Field(description="Voice used for synthesis")
    status: str = "success"
    timestamp: str = ""


class STTRequest(BaseModel):
    """Simple speech-to-text request."""
    audio: str = Field(description="Base64-encoded audio data")
    audio_format: str = Field(default="wav",
        description="Audio format: wav, flac, mp3, ogg, webm")


class STTResponse(BaseModel):
    """Speech-to-text response."""
    text: str = Field(description="Transcribed text")
    status: str = "success"
    timestamp: str = ""


# ── Wearable / Health Models ─────────────────────────────


class SleepRecord(BaseModel):
    """Sleep data from wearable or manual entry."""
    duration_hours: float = 0.0               # total sleep duration
    quality: str = "unknown"                  # "poor", "fair", "good", "excellent"
    deep_sleep_minutes: int = 0
    rem_sleep_minutes: int = 0
    light_sleep_minutes: int = 0
    awakenings: int = 0
    sleep_start: Optional[str] = None         # ISO timestamp
    sleep_end: Optional[str] = None           # ISO timestamp


class ActivityRecord(BaseModel):
    """Activity data from wearable or manual entry."""
    steps: int = 0
    active_minutes: int = 0
    calories_burned: int = 0
    distance_km: float = 0.0
    exercise_type: Optional[str] = None       # "walking", "running", "yoga", etc.


class DailyHealthMetrics(BaseModel):
    """One day of health data combining sleep + activity."""
    user_id: str
    date: str                                 # "YYYY-MM-DD"
    sleep: Optional[SleepRecord] = None
    activity: Optional[ActivityRecord] = None
    source: str = "manual"                    # "apple_health", "google_fit", "manual"
    synced_at: str = ""                       # ISO timestamp


class HealthSyncRequest(BaseModel):
    """Batch sync request from client (Flutter app)."""
    user_id: str
    metrics: List[DailyHealthMetrics]         # batch of daily records
    source: str = "manual"


class HealthSummary(BaseModel):
    """Aggregated health summary with trends."""
    user_id: str
    period_days: int = 7
    avg_sleep_hours: float = 0.0
    avg_steps: int = 0
    avg_active_minutes: int = 0
    sleep_trend: str = "stable"               # "improving", "declining", "stable"
    activity_trend: str = "stable"            # "improving", "declining", "stable"
    total_records: int = 0
    status: str = "success"
    timestamp: str = ""


# ── Fine-Tuning & Model Performance (Phase 18) ──────────


class TrainingExampleSource(str, Enum):
    """Source of a training example."""
    CHAT_SESSION = "chat_session"
    EXERCISE_SESSION = "exercise_session"


class TrainingExample(BaseModel):
    """A single training example extracted for fine-tuning."""
    example_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str
    source_type: TrainingExampleSource = TrainingExampleSource.CHAT_SESSION
    source_id: str = Field(default="",
        description="session_id this was extracted from")
    messages: List[dict] = Field(default_factory=list,
        description="OpenAI fine-tuning format: [{role, content}, ...]")
    quality_signals: dict = Field(default_factory=dict,
        description="Feedback rating, helpfulness score, mood_delta, etc.")
    included_in_job: str = Field(default="",
        description="Fine-tuning job ID if used, empty if unused")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class FineTuningJobStatus(str, Enum):
    """Status of a fine-tuning job."""
    PENDING = "pending"
    VALIDATING = "validating_files"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FineTuningJobRecord(BaseModel):
    """Record of a fine-tuning job submitted to OpenAI."""
    job_id: str = Field(description="OpenAI fine-tuning job ID (ftjob-...)")
    base_model: str = Field(default="gpt-4o-mini")
    fine_tuned_model_id: str = Field(default="",
        description="Resulting model ID after training (ft:gpt-4o-mini:...)")
    status: FineTuningJobStatus = FineTuningJobStatus.PENDING
    training_file_id: str = Field(default="",
        description="OpenAI file ID for uploaded training JSONL")
    training_examples_count: int = Field(default=0)
    hyperparameters: dict = Field(default_factory=dict,
        description="e.g. {'n_epochs': 3}")
    error_message: str = Field(default="")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    finished_at: str = Field(default="")
    created_by: str = Field(default="system")


class ModelPerformanceRecord(BaseModel):
    """Per-response performance tracking for A/B comparison."""
    record_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str
    session_id: str
    model_used: str = Field(description="Model ID that generated this response")
    is_fine_tuned: bool = Field(default=False)
    ab_group: str = Field(default="base",
        description="'base' or 'fine_tuned'")
    response_length: int = Field(default=0,
        description="Character count of the assistant response")
    feedback_rating: str = Field(default="",
        description="If user gave feedback on this response")
    detected_emotion: str = Field(default="")
    detected_intent: str = Field(default="")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class FineTuningStatsResponse(BaseModel):
    """API response for A/B performance comparison stats."""
    total_responses_base: int = 0
    total_responses_ft: int = 0
    helpful_rate_base: float = Field(default=0.0,
        description="Fraction of base model responses rated 'helpful'")
    helpful_rate_ft: float = Field(default=0.0,
        description="Fraction of fine-tuned model responses rated 'helpful'")
    avg_response_length_base: float = 0.0
    avg_response_length_ft: float = 0.0
    active_model_id: str = Field(default="",
        description="Currently active fine-tuned model ID")
    ab_split_percent: int = 50
    status: str = "success"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExtractTrainingDataRequest(BaseModel):
    """Request to extract training data from chat sessions."""
    min_feedback_rating: str = Field(default="helpful",
        description="Minimum rating: 'helpful' or 'neutral'")
    min_helpfulness_score: int = Field(default=4, ge=1, le=5,
        description="Minimum exercise outcome helpfulness to include")
    max_examples: int = Field(default=1000, ge=10, le=10000,
        description="Maximum number of examples to extract")


class SubmitFineTuningJobRequest(BaseModel):
    """Request to submit a fine-tuning job to OpenAI."""
    base_model: str = Field(default="gpt-4o-mini")
    n_epochs: int = Field(default=3, ge=1, le=10)
    suffix: str = Field(default="lucille",
        description="Suffix for the fine-tuned model name")


# ── Monitoring Dashboard (Phase 19) ─────────────────────────


class DashboardSystemMetrics(BaseModel):
    """System health and request metrics for the monitoring dashboard."""
    uptime_seconds: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
    overall_error_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    cache_hit_rate: float = 0.0
    cache_size: int = 0
    memory_rss_mb: float = 0.0
    memory_percent: float = 0.0
    environment: str = ""
    model: str = ""
    top_endpoints: List[dict] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class DashboardEngagementMetrics(BaseModel):
    """User engagement metrics aggregated across all users."""
    total_users: int = 0
    active_users_today: int = 0
    total_sessions: int = 0
    total_messages_today: int = 0
    new_users_7d: int = 0
    avg_sessions_per_user: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class DashboardTherapyMetrics(BaseModel):
    """Aggregated therapy effectiveness metrics."""
    total_exercise_outcomes: int = 0
    avg_helpfulness: float = 0.0
    exercise_completion_rate: float = 0.0
    avg_mood_improvement: float = 0.0
    modality_breakdown: dict = Field(default_factory=dict)
    top_exercises: List[dict] = Field(default_factory=list)
    mood_improvement_distribution: dict = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class DashboardSafetyMetrics(BaseModel):
    """Safety event overview aggregated across all users."""
    total_events: int = 0
    events_by_risk_level: dict = Field(default_factory=dict)
    events_by_type: dict = Field(default_factory=dict)
    recent_critical_events: List[dict] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class DashboardModelMetrics(BaseModel):
    """A/B model performance metrics for the monitoring dashboard."""
    ft_enabled: bool = False
    active_model_id: str = ""
    ab_split_percent: int = 50
    base_total: int = 0
    base_helpful_rate: float = 0.0
    base_avg_length: float = 0.0
    ft_total: int = 0
    ft_helpful_rate: float = 0.0
    ft_avg_length: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class DashboardRLMetrics(BaseModel):
    """RL / Thompson Sampling bandit overview for the monitoring dashboard."""
    rl_enabled: bool = False
    total_users_with_bandit_state: int = 0
    total_arms: int = 0
    top_arms_by_group: dict = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class DashboardAllMetrics(BaseModel):
    """Combined response containing all dashboard sections."""
    system: DashboardSystemMetrics = Field(default_factory=DashboardSystemMetrics)
    engagement: DashboardEngagementMetrics = Field(default_factory=DashboardEngagementMetrics)
    therapy: DashboardTherapyMetrics = Field(default_factory=DashboardTherapyMetrics)
    safety: DashboardSafetyMetrics = Field(default_factory=DashboardSafetyMetrics)
    models: DashboardModelMetrics = Field(default_factory=DashboardModelMetrics)
    rl: DashboardRLMetrics = Field(default_factory=DashboardRLMetrics)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Phase 20: Human Escalation & Annual Reviews ───────


class EscalationTriggerType(str, Enum):
    """What triggered the escalation."""
    CRITICAL_SAFETY = "critical_safety"
    HIGH_DEPENDENCY = "high_dependency"
    REPEATED_HIGH_RISK = "repeated_high_risk"
    MANUAL = "manual"


class EscalationPriority(str, Enum):
    """Escalation priority level."""
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"


class EscalationStatus(str, Enum):
    """Escalation workflow status."""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class EscalationEvent(BaseModel):
    """A human escalation ticket created when safety/dependency thresholds are crossed."""
    escalation_id: str = Field(default_factory=lambda: "")
    user_id: str = ""
    reason: str = ""
    trigger_type: str = EscalationTriggerType.MANUAL.value
    priority: str = EscalationPriority.NORMAL.value
    status: str = EscalationStatus.PENDING.value
    safety_event_ids: List[str] = Field(default_factory=list)
    dependency_score: Optional[int] = None
    notes: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None


class UpdateEscalationRequest(BaseModel):
    """Request to update an escalation ticket."""
    status: Optional[str] = None
    notes: Optional[str] = None
    resolved_by: Optional[str] = None


class EscalationStats(BaseModel):
    """Aggregate statistics for the escalation queue."""
    total_pending: int = 0
    total_acknowledged: int = 0
    total_in_progress: int = 0
    total_resolved: int = 0
    total_dismissed: int = 0
    by_priority: dict = Field(default_factory=dict)
    by_trigger_type: dict = Field(default_factory=dict)
    avg_resolution_hours: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class GenerateReviewRequest(BaseModel):
    """Request to generate a periodic review for a user."""
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class AnnualReview(BaseModel):
    """Comprehensive periodic review aggregating all user data with LLM narrative."""
    review_id: str = Field(default_factory=lambda: "")
    user_id: str = ""
    review_period_start: str = ""
    review_period_end: str = ""
    progress_summary: dict = Field(default_factory=dict)
    effectiveness_summary: dict = Field(default_factory=dict)
    safety_summary: dict = Field(default_factory=dict)
    health_summary: dict = Field(default_factory=dict)
    engagement_summary: dict = Field(default_factory=dict)
    rl_insights: dict = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    narrative: str = ""
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Assessment System (Phase 21) ─────────────────────────

class AssessmentType(str, Enum):
    PHQ9 = "phq9"
    GAD7 = "gad7"
    WHO5 = "who5"


class AssessmentSeverity(str, Enum):
    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    MODERATELY_SEVERE = "moderately_severe"
    SEVERE = "severe"


class AssessmentSessionStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class AssessmentQuestion(BaseModel):
    """A single item from a validated assessment instrument."""
    index: int = Field(description="0-based position in the instrument")
    text: str = Field(description="Exact published question wording")
    min_value: int = Field(default=0, description="Minimum allowed response value")
    max_value: int = Field(description="Maximum allowed response value (3 for PHQ-9/GAD-7, 5 for WHO-5)")
    value_labels: dict = Field(default_factory=dict, description="Maps int value to label text")


class AssessmentAnswer(BaseModel):
    """A single submitted answer to an assessment question."""
    question_index: int = Field(description="Index of the question answered")
    value: int = Field(description="User's response value")
    answered_at: str = Field(default_factory=lambda: datetime.now().isoformat())


ASSESSMENT_DISCLAIMER = (
    "This score is from a standardized screening tool and is not a clinical diagnosis. "
    "Please consult a healthcare professional for clinical evaluation."
)


class AssessmentResult(BaseModel):
    """Computed scores after completing a validated assessment."""
    assessment_type: AssessmentType
    raw_score: int = Field(description="Sum of all item values")
    scaled_score: Optional[int] = Field(default=None, description="Scaled score (WHO-5 only: raw x 4)")
    severity: AssessmentSeverity
    severity_label: str = Field(description="Human-readable severity, e.g. 'Moderate Depression'")
    concern_flags: List[str] = Field(default_factory=list, description="Flags like 'elevated_depression', 'self_harm_ideation'")
    disclaimer: str = Field(default=ASSESSMENT_DISCLAIMER)


class AssessmentSession(BaseModel):
    """Tracks one assessment attempt through a validated instrument."""
    session_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str
    assessment_type: AssessmentType
    status: AssessmentSessionStatus = AssessmentSessionStatus.IN_PROGRESS
    answers: List[AssessmentAnswer] = Field(default_factory=list)
    current_question_index: int = Field(default=0)
    total_questions: int
    result: Optional[AssessmentResult] = None
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    safety_flagged: bool = Field(default=False, description="True if PHQ-9 Q9 self-harm ideation > 0")


class WellnessScore(BaseModel):
    """Composite wellness view: WHO-5 as primary score, PHQ-9/GAD-7 as breakdowns."""
    user_id: str
    overall_score: Optional[int] = Field(default=None, description="WHO-5 scaled score 0-100, None if no WHO-5 yet")
    overall_label: str = Field(default="No assessment yet")
    phq9_score: Optional[int] = None
    phq9_severity: Optional[str] = None
    gad7_score: Optional[int] = None
    gad7_severity: Optional[str] = None
    who5_raw: Optional[int] = None
    who5_scaled: Optional[int] = None
    concern_flags: List[str] = Field(default_factory=list)
    last_assessment_dates: dict = Field(default_factory=dict, description="Maps assessment type to last completion ISO date")
    disclaimer: str = Field(default=ASSESSMENT_DISCLAIMER)
    computed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Self-Care Score Models (Phase 22) ─────────────────────

SELFCARE_DISCLAIMER = (
    "This score reflects your engagement with self-care activities, "
    "not your clinical mental health status. It is not a medical assessment."
)


class SelfCareScoreCategory(BaseModel):
    """One dimension of the Self-Care Score."""
    name: str = Field(description="Category name, e.g. 'Mood Stability'")
    score: float = Field(default=0.0, ge=0.0, le=100.0, description="Raw score for this category (0-100)")
    weight: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight applied (0.0-1.0)")
    weighted_score: float = Field(default=0.0, ge=0.0, description="score * weight contribution to total")
    detail: str = Field(default="", description="Human-readable explanation")


class SelfCareScore(BaseModel):
    """
    Composite engagement and self-care score (0-100).

    NOT a clinical assessment. Measures how consistently the user engages
    with self-care activities (mood logging, exercises, tasks, streaks).
    """
    user_id: str
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    categories: List[SelfCareScoreCategory] = Field(default_factory=list)
    trend: str = Field(default="stable", description="'improving', 'stable', or 'declining'")
    burnout_flag: bool = Field(default=False)
    burnout_detail: str = Field(default="")
    insight_text: str = Field(default="", description="Lucille's personalized insight")
    data_sufficiency: str = Field(default="insufficient", description="'insufficient', 'partial', or 'full'")
    disclaimer: str = Field(default=SELFCARE_DISCLAIMER)
    computed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Assessment Request/Response Models ────────────────────

class StartAssessmentRequest(BaseModel):
    """Request to start a validated assessment."""
    assessment_type: AssessmentType


class StartAssessmentResponse(BaseModel):
    """Response when starting an assessment."""
    session_id: str
    assessment_type: str
    total_questions: int
    first_question: AssessmentQuestion
    status: str = "success"
    timestamp: str = ""


class SubmitAnswerRequest(BaseModel):
    """Request to submit an answer to the current assessment question."""
    value: int = Field(ge=0, le=5, description="Response value (0-3 for PHQ-9/GAD-7, 0-5 for WHO-5)")


class SubmitAnswerResponse(BaseModel):
    """Response after submitting an assessment answer."""
    session_id: str
    question_answered: int = Field(description="Index of the question just answered")
    next_question: Optional[AssessmentQuestion] = None
    is_complete: bool = Field(default=False, description="True if all questions answered")
    safety_notice: Optional[str] = Field(default=None, description="Crisis resources if safety flagged")
    status: str = "success"
    timestamp: str = ""


class CompleteAssessmentResponse(BaseModel):
    """Response after completing and scoring an assessment."""
    session_id: str
    result: AssessmentResult
    safety_flagged: bool = False
    safety_resources: Optional[List[dict]] = Field(default=None, description="Crisis resources if concern flags present")
    status: str = "success"
    timestamp: str = ""


class AssessmentHistoryResponse(BaseModel):
    """Response with past assessment results."""
    user_id: str
    assessments: List[dict] = Field(default_factory=list)
    count: int = 0
    status: str = "success"
    timestamp: str = ""
