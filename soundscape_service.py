"""
LucilleLLM - Soundscape & Audio Engine

Provides a curated catalog of ambient audio soundscapes for relaxation,
focus, and exercise accompaniment. Features emotion-based recommendations,
exercise-soundscape pairing, session tracking, and prompt formatting.

Follows the singleton pattern from other services.
"""

import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from firebase_service import get_firebase_service
from storage_service import get_storage_service
from models import (
    SoundscapeCategory,
    SoundscapeRecommendation,
    SoundscapeSession,
    SoundscapeTemplate,
)

logger = logging.getLogger(__name__)


# ── Soundscape Template Registry ─────────────────────────

SOUNDSCAPE_REGISTRY: Dict[str, SoundscapeTemplate] = {}


def _register(ss: SoundscapeTemplate) -> None:
    SOUNDSCAPE_REGISTRY[ss.soundscape_id] = ss


# ── Nature Soundscapes (4) ───────────────────────────────

_register(SoundscapeTemplate(
    soundscape_id="nature_rain",
    category=SoundscapeCategory.NATURE,
    title="Gentle Rain",
    description="Steady rainfall with occasional distant thunder. Perfect for calming an overactive mind.",
    duration_seconds=300,
    target_emotions=["anxious", "overwhelmed", "sad", "angry"],
    target_contexts=["relaxation", "sleep", "reflection"],
    icon="🌧️",
))

_register(SoundscapeTemplate(
    soundscape_id="nature_ocean",
    category=SoundscapeCategory.NATURE,
    title="Ocean Waves",
    description="Rhythmic ocean waves on a sandy beach. A timeless sound for deep relaxation.",
    duration_seconds=300,
    target_emotions=["anxious", "overwhelmed", "lonely", "sad"],
    target_contexts=["relaxation", "sleep", "meditation"],
    icon="🌊",
))

_register(SoundscapeTemplate(
    soundscape_id="nature_forest",
    category=SoundscapeCategory.NATURE,
    title="Forest Ambiance",
    description="Birds singing, rustling leaves, and a gentle breeze through the trees.",
    duration_seconds=300,
    target_emotions=["anxious", "lonely", "sad", "overwhelmed"],
    target_contexts=["relaxation", "reflection", "grounding"],
    icon="🌲",
))

_register(SoundscapeTemplate(
    soundscape_id="nature_stream",
    category=SoundscapeCategory.NATURE,
    title="Mountain Stream",
    description="Clear water flowing over smooth rocks with birds in the background.",
    duration_seconds=300,
    target_emotions=["anxious", "overwhelmed", "angry"],
    target_contexts=["focus", "relaxation", "reflection"],
    icon="🏞️",
))

# ── Ambient Soundscapes (3) ──────────────────────────────

_register(SoundscapeTemplate(
    soundscape_id="ambient_cafe",
    category=SoundscapeCategory.AMBIENT,
    title="Coffee Shop",
    description="Soft cafe sounds with murmured conversation and clinking cups. Great for comfortable focus.",
    duration_seconds=300,
    target_emotions=["lonely", "sad"],
    target_contexts=["focus", "companionship", "casual"],
    icon="☕",
))

_register(SoundscapeTemplate(
    soundscape_id="ambient_fireplace",
    category=SoundscapeCategory.AMBIENT,
    title="Fireplace Crackling",
    description="Warm fireplace with gentle crackling and occasional pops. Cozy and soothing.",
    duration_seconds=300,
    target_emotions=["anxious", "lonely", "sad", "fearful"],
    target_contexts=["relaxation", "comfort", "sleep"],
    icon="🔥",
))

_register(SoundscapeTemplate(
    soundscape_id="ambient_library",
    category=SoundscapeCategory.AMBIENT,
    title="Library Quiet",
    description="Very soft ambient sounds with distant page turns and muffled footsteps. Deep focus environment.",
    duration_seconds=300,
    target_emotions=["overwhelmed", "anxious"],
    target_contexts=["focus", "study", "reflection"],
    icon="📚",
))

# ── Meditation Soundscapes (4) ───────────────────────────

_register(SoundscapeTemplate(
    soundscape_id="meditation_bowls",
    category=SoundscapeCategory.MEDITATION,
    title="Singing Bowls",
    description="Tibetan singing bowls with rich harmonic overtones. Ideal for meditation and mindfulness.",
    duration_seconds=300,
    target_emotions=["anxious", "overwhelmed", "angry", "fearful"],
    target_contexts=["meditation", "mindfulness", "grounding"],
    icon="🔔",
))

_register(SoundscapeTemplate(
    soundscape_id="meditation_breath",
    category=SoundscapeCategory.MEDITATION,
    title="Breath Guide",
    description="Soft tones guiding inhale and exhale rhythms. Supports breathing exercises.",
    duration_seconds=300,
    target_emotions=["anxious", "overwhelmed", "fearful", "angry"],
    target_contexts=["breathing", "meditation", "grounding"],
    icon="🌬️",
))

_register(SoundscapeTemplate(
    soundscape_id="meditation_chimes",
    category=SoundscapeCategory.MEDITATION,
    title="Wind Chimes",
    description="Gentle wind chimes with a soft breeze. Light and uplifting meditation background.",
    duration_seconds=300,
    target_emotions=["sad", "lonely", "hopeless"],
    target_contexts=["meditation", "relaxation", "reflection"],
    icon="🎐",
))

_register(SoundscapeTemplate(
    soundscape_id="meditation_om",
    category=SoundscapeCategory.MEDITATION,
    title="Om Resonance",
    description="Deep om chanting loop with harmonic resonance. For deep meditative states.",
    duration_seconds=300,
    target_emotions=["anxious", "overwhelmed", "fearful"],
    target_contexts=["meditation", "grounding", "spiritual"],
    icon="🕉️",
))

# ── Binaural Soundscapes (3) ─────────────────────────────

_register(SoundscapeTemplate(
    soundscape_id="binaural_alpha",
    category=SoundscapeCategory.BINAURAL,
    title="Alpha Waves (10Hz)",
    description="Binaural beats at 10Hz alpha frequency. Promotes calm alertness and relaxation.",
    duration_seconds=300,
    target_emotions=["anxious", "overwhelmed", "angry"],
    target_contexts=["relaxation", "focus", "stress-relief"],
    icon="🧠",
))

_register(SoundscapeTemplate(
    soundscape_id="binaural_theta",
    category=SoundscapeCategory.BINAURAL,
    title="Theta Waves (6Hz)",
    description="Binaural beats at 6Hz theta frequency. Supports meditation and creative thinking.",
    duration_seconds=300,
    target_emotions=["anxious", "sad", "hopeless"],
    target_contexts=["meditation", "creativity", "deep-relaxation"],
    icon="🌀",
))

_register(SoundscapeTemplate(
    soundscape_id="binaural_delta",
    category=SoundscapeCategory.BINAURAL,
    title="Delta Waves (2Hz)",
    description="Binaural beats at 2Hz delta frequency. For deep relaxation and sleep preparation.",
    duration_seconds=300,
    target_emotions=["anxious", "overwhelmed", "fearful"],
    target_contexts=["sleep", "deep-relaxation", "recovery"],
    icon="💤",
))

# ── Music Soundscapes (3) ────────────────────────────────

_register(SoundscapeTemplate(
    soundscape_id="music_piano",
    category=SoundscapeCategory.MUSIC,
    title="Soft Piano",
    description="Gentle piano melodies with soft reverb. Emotionally soothing and reflective.",
    duration_seconds=300,
    target_emotions=["sad", "lonely", "hopeless", "grateful"],
    target_contexts=["reflection", "relaxation", "emotional-processing"],
    icon="🎹",
))

_register(SoundscapeTemplate(
    soundscape_id="music_ambient_electronic",
    category=SoundscapeCategory.MUSIC,
    title="Ambient Electronic",
    description="Soft synthesizer pads with gentle evolving textures. Modern and calming.",
    duration_seconds=300,
    target_emotions=["anxious", "overwhelmed", "sad"],
    target_contexts=["focus", "relaxation", "study"],
    icon="🎧",
))

_register(SoundscapeTemplate(
    soundscape_id="music_acoustic",
    category=SoundscapeCategory.MUSIC,
    title="Acoustic Guitar",
    description="Gentle fingerpicked acoustic guitar. Warm, uplifting, and grounding.",
    duration_seconds=300,
    target_emotions=["sad", "lonely", "hopeless", "happy"],
    target_contexts=["relaxation", "uplifting", "reflection"],
    icon="🎸",
))


# ── Exercise-Soundscape Map ──────────────────────────────

EXERCISE_SOUNDSCAPE_MAP: Dict[str, str] = {
    "cbt_thought_record": "nature_rain",
    "cbt_cognitive_distortions": "ambient_library",
    "cbt_behavioral_activation": "music_acoustic",
    "act_defusion": "meditation_bowls",
    "act_values_compass": "nature_forest",
    "dbt_distress_tolerance": "binaural_alpha",
    "dbt_opposite_action": "music_piano",
    "dbt_mindfulness_observe": "meditation_breath",
    "mi_decisional_balance": "ambient_cafe",
    "mi_scaling_question": "nature_stream",
}


# ── Emotion-Soundscape Scoring Hints ─────────────────────

_EMOTION_CONTEXT_MAP = {
    "anxious": ["relaxation", "breathing", "grounding"],
    "overwhelmed": ["relaxation", "focus", "grounding"],
    "sad": ["comfort", "reflection", "companionship"],
    "angry": ["relaxation", "grounding", "stress-relief"],
    "fearful": ["grounding", "comfort", "relaxation"],
    "hopeless": ["uplifting", "reflection", "meditation"],
    "lonely": ["companionship", "comfort", "relaxation"],
    "happy": ["uplifting", "relaxation", "meditation"],
    "grateful": ["reflection", "meditation", "relaxation"],
}


class SoundscapeService:
    """
    Service for soundscape catalog, recommendations, and session tracking.

    Firestore structure:
        soundscape_sessions/{user_id}/sessions/{session_id}
    """

    COLLECTION = "soundscape_sessions"

    def __init__(self):
        self._firebase = get_firebase_service()
        self._storage = get_storage_service()

    @property
    def db(self):
        return self._firebase.db

    # ── Catalog Operations ────────────────────────────────

    def _populate_audio_url(self, ss: SoundscapeTemplate) -> SoundscapeTemplate:
        """Return a copy of the template with audio_url populated from GCS."""
        if self._storage.is_configured and not ss.audio_url:
            url = self._storage.get_audio_url(ss.soundscape_id)
            if url:
                return ss.model_copy(update={"audio_url": url})
        return ss

    def list_soundscapes(
        self, category: Optional[str] = None
    ) -> List[SoundscapeTemplate]:
        """List all soundscapes, optionally filtered by category."""
        if category:
            templates = [
                ss for ss in SOUNDSCAPE_REGISTRY.values()
                if ss.category.value == category.lower()
            ]
        else:
            templates = list(SOUNDSCAPE_REGISTRY.values())
        return [self._populate_audio_url(ss) for ss in templates]

    def get_soundscape(
        self, soundscape_id: str
    ) -> Optional[SoundscapeTemplate]:
        """Get a single soundscape by ID."""
        ss = SOUNDSCAPE_REGISTRY.get(soundscape_id)
        if ss is not None:
            ss = self._populate_audio_url(ss)
        return ss

    # ── Recommendation Engine ─────────────────────────────

    def recommend_soundscapes(
        self,
        emotion: str,
        intent: str,
        exercise_id: Optional[str] = None,
        limit: int = 3,
    ) -> List[SoundscapeRecommendation]:
        """
        Recommend soundscapes based on emotion, intent, and optional exercise.

        Scoring:
        - Emotion match in target_emotions: +3
        - Context match (emotion-derived contexts vs target_contexts): +2
        - Exercise map match: +5
        """
        scored = []
        emotion_contexts = _EMOTION_CONTEXT_MAP.get(emotion, [])

        for ss in SOUNDSCAPE_REGISTRY.values():
            score = 0
            reasons = []

            # Emotion match
            if emotion in ss.target_emotions:
                score += 3
                reasons.append(f"matches your {emotion} mood")

            # Context match
            context_overlap = set(emotion_contexts) & set(ss.target_contexts)
            if context_overlap:
                score += 2
                reasons.append(
                    f"good for {', '.join(context_overlap)}"
                )

            # Exercise map match
            if exercise_id and EXERCISE_SOUNDSCAPE_MAP.get(exercise_id) == ss.soundscape_id:
                score += 5
                reasons.append("recommended for this exercise")

            if score > 0:
                reason = "; ".join(reasons) if reasons else "general recommendation"
                scored.append((score, ss, reason))

        # Sort by score descending, then ensure category diversity
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        seen_categories = set()
        for score, ss, reason in scored:
            if len(results) >= limit:
                break
            # Diversity bonus: prefer unseen categories when scores are close
            if ss.category.value not in seen_categories:
                seen_categories.add(ss.category.value)
            # Populate audio_url from GCS if available
            audio_url = ss.audio_url
            if self._storage.is_configured and not audio_url:
                audio_url = self._storage.get_audio_url(ss.soundscape_id)

            results.append(SoundscapeRecommendation(
                soundscape_id=ss.soundscape_id,
                category=ss.category.value,
                title=ss.title,
                description=ss.description,
                reason=reason.capitalize(),
                audio_url=audio_url,
                icon=ss.icon,
            ))

        return results

    # ── Session Management ────────────────────────────────

    def start_session(
        self, user_id: str, soundscape_id: str
    ) -> Optional[SoundscapeSession]:
        """Start a soundscape listening session."""
        ss = self.get_soundscape(soundscape_id)
        if ss is None:
            return None

        session = SoundscapeSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            soundscape_id=soundscape_id,
            title=ss.title,
            category=ss.category.value,
        )

        if self.db is not None:
            try:
                doc_ref = (
                    self.db.collection(self.COLLECTION)
                    .document(user_id)
                    .collection("sessions")
                    .document(session.session_id)
                )
                doc_ref.set(session.model_dump())
                logger.info(
                    f"Started soundscape session {session.session_id} "
                    f"({ss.title}) for user {user_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to store soundscape session: {e}")

        return session

    def stop_session(
        self, user_id: str, session_id: str
    ) -> Optional[SoundscapeSession]:
        """Stop a soundscape session, computing duration."""
        if self.db is None:
            return None
        try:
            doc_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("sessions")
                .document(session_id)
            )
            doc = doc_ref.get()
            if not doc.exists:
                return None

            data = doc.to_dict()
            if data.get("stopped_at") is not None:
                # Already stopped
                return SoundscapeSession(**data)

            now = datetime.now()
            started_at = datetime.fromisoformat(data["started_at"])
            duration_seconds = int((now - started_at).total_seconds())

            doc_ref.update({
                "stopped_at": now.isoformat(),
                "duration_seconds": duration_seconds,
            })

            data["stopped_at"] = now.isoformat()
            data["duration_seconds"] = duration_seconds
            logger.info(
                f"Stopped soundscape session {session_id} "
                f"(duration: {duration_seconds}s) for user {user_id}"
            )
            return SoundscapeSession(**data)

        except Exception as e:
            logger.warning(f"Failed to stop soundscape session: {e}")
            return None

    def get_active_session(
        self, user_id: str
    ) -> Optional[SoundscapeSession]:
        """Get the currently active soundscape session (stopped_at is None)."""
        if self.db is None:
            return None
        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("sessions")
            )
            query = col_ref.where(
                "stopped_at", "==", None
            ).limit(1)
            docs = list(query.stream())
            if docs:
                return SoundscapeSession(**docs[0].to_dict())
            return None
        except Exception as e:
            logger.warning(f"Failed to get active soundscape session: {e}")
            return None

    def get_session_history(
        self, user_id: str, limit: int = 20
    ) -> List[dict]:
        """Get recent soundscape sessions for a user."""
        if self.db is None:
            return []
        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("sessions")
            )
            query = col_ref.order_by(
                "started_at", direction="DESCENDING"
            ).limit(limit)
            return [doc.to_dict() for doc in query.stream()]
        except Exception as e:
            logger.warning(f"Failed to get soundscape history: {e}")
            return []

    # ── Prompt Formatting ─────────────────────────────────

    def format_suggestion_for_prompt(
        self, emotion: str, intensity: int
    ) -> str:
        """
        Format a soundscape suggestion for the system prompt.
        Only called when intensity >= 5.
        """
        recommendations = self.recommend_soundscapes(
            emotion=emotion, intent="", limit=2
        )
        if not recommendations:
            return ""

        lines = ["--- SOUNDSCAPE SUGGESTION ---"]
        lines.append(
            f"The user seems to be feeling {emotion} (intensity {intensity}/10). "
            "You may gently suggest trying one of these ambient soundscapes "
            "to help them feel more comfortable:"
        )
        for rec in recommendations:
            lines.append(
                f"  - {rec.icon} {rec.title} ({rec.category}): {rec.reason}"
            )
        lines.append(
            "Only suggest a soundscape if it feels natural in the conversation. "
            "Don't force it."
        )
        lines.append("--- END SOUNDSCAPE SUGGESTION ---")
        return "\n".join(lines)

    # ── Audio Status ──────────────────────────────────────

    def get_audio_status(self) -> dict:
        """
        Check which soundscapes have audio files uploaded to GCS.

        Returns:
            Dict of {soundscape_id: bool} indicating audio availability.
        """
        status = {}
        for soundscape_id in SOUNDSCAPE_REGISTRY:
            if self._storage.is_configured:
                status[soundscape_id] = self._storage.audio_file_exists(
                    soundscape_id
                )
            else:
                status[soundscape_id] = False
        return status


# ── Singleton ─────────────────────────────────────────

_soundscape_service: Optional[SoundscapeService] = None


def get_soundscape_service() -> SoundscapeService:
    """Get or create SoundscapeService singleton."""
    global _soundscape_service
    if _soundscape_service is None:
        _soundscape_service = SoundscapeService()
    return _soundscape_service
