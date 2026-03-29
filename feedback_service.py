"""
LucilleLLM - Feedback & Closed-Loop Service

Stores user feedback on chat responses and exercise outcomes.
Computes effectiveness profiles at query time for adaptive
recommendations and prompt insights.

Follows the singleton pattern from other services.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import List, Optional

from firebase_service import get_firebase_service
from therapy_service import EXERCISE_REGISTRY
from models import (
    EffectivenessProfile,
    ExerciseOutcome,
    FeedbackRating,
    ResponseFeedback,
)

logger = logging.getLogger(__name__)


class FeedbackService:
    """
    Service for feedback storage, retrieval, and effectiveness computation.

    Firestore structure:
        feedback/{user_id}/response_feedback/{feedback_id}
        feedback/{user_id}/exercise_outcomes/{outcome_id}
    """

    COLLECTION = "feedback"

    def __init__(self):
        self._firebase = get_firebase_service()

    @property
    def db(self):
        return self._firebase.db

    # ── Store Operations ─────────────────────────────────

    def store_response_feedback(
        self, user_id: str, feedback: ResponseFeedback
    ) -> Optional[str]:
        """Store a chat response feedback entry. Returns feedback_id or None."""
        if self.db is None:
            return None
        try:
            doc_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("response_feedback")
                .document(feedback.feedback_id)
            )
            doc_ref.set(feedback.model_dump())
            logger.info(
                f"Stored response feedback {feedback.feedback_id} "
                f"for user {user_id}"
            )
            return feedback.feedback_id
        except Exception as e:
            logger.warning(f"Failed to store response feedback: {e}")
            return None

    def store_exercise_outcome(
        self, user_id: str, outcome: ExerciseOutcome
    ) -> Optional[str]:
        """Store an exercise outcome entry. Returns outcome_id or None.
        Also updates RL bandit state if RL is enabled (Phase 17).
        """
        if self.db is None:
            return None
        try:
            doc_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("exercise_outcomes")
                .document(outcome.outcome_id)
            )
            doc_ref.set(outcome.model_dump())
            logger.info(
                f"Stored exercise outcome {outcome.outcome_id} "
                f"for user {user_id}"
            )

            # Phase 17: Update RL bandit state
            self._update_rl_reward(user_id, outcome)

            return outcome.outcome_id
        except Exception as e:
            logger.warning(f"Failed to store exercise outcome: {e}")
            return None

    def _update_rl_reward(
        self, user_id: str, outcome: ExerciseOutcome
    ) -> None:
        """Update Thompson Sampling bandit state from exercise outcome.
        Fails silently to avoid blocking the feedback flow."""
        try:
            from config import get_config
            config = get_config()
            if not config.RL_ENABLED:
                return

            from rl_service import get_rl_service
            rl_svc = get_rl_service()

            # Determine the emotion context for this outcome
            emotion = self._get_emotion_for_outcome(user_id)

            rl_svc.update_reward(
                user_id=user_id,
                exercise_id=outcome.exercise_id,
                emotion=emotion,
                helpfulness=outcome.helpfulness,
                mood_before=outcome.mood_before,
                mood_after=outcome.mood_after,
                would_repeat=outcome.would_repeat,
            )
            logger.info(
                f"Updated RL bandit state for user {user_id}, "
                f"exercise {outcome.exercise_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to update RL reward: {e}")

    def _get_emotion_for_outcome(self, user_id: str) -> str:
        """
        Determine the emotion context for a bandit reward update.
        Looks at the user's most recent mood entry, or defaults to 'neutral'.
        """
        try:
            from user_service import get_user_service
            user_svc = get_user_service()
            profile_data = user_svc.get_user_profile(user_id)
            if profile_data:
                mood_history = profile_data.get("affective", {}).get(
                    "mood_history", []
                )
                if mood_history:
                    return mood_history[-1].get("mood", "neutral")
        except Exception:
            pass
        return "neutral"

    # ── Query Operations ──────────────────────────────────

    def get_response_feedback(
        self, user_id: str, limit: int = 20
    ) -> List[dict]:
        """Get recent response feedback for a user."""
        if self.db is None:
            return []
        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("response_feedback")
            )
            query = col_ref.order_by(
                "created_at", direction="DESCENDING"
            ).limit(limit)
            return [doc.to_dict() for doc in query.stream()]
        except Exception as e:
            logger.warning(f"Failed to get response feedback: {e}")
            return []

    def get_exercise_outcomes(
        self, user_id: str, limit: int = 20
    ) -> List[dict]:
        """Get recent exercise outcomes for a user."""
        if self.db is None:
            return []
        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("exercise_outcomes")
            )
            query = col_ref.order_by(
                "created_at", direction="DESCENDING"
            ).limit(limit)
            return [doc.to_dict() for doc in query.stream()]
        except Exception as e:
            logger.warning(f"Failed to get exercise outcomes: {e}")
            return []

    # ── Effectiveness Computation ─────────────────────────

    def compute_effectiveness(self, user_id: str) -> EffectivenessProfile:
        """
        Compute per-modality and per-exercise effectiveness from
        stored outcome data. All computed at query time.
        """
        profile = EffectivenessProfile(user_id=user_id)

        if self.db is None:
            return profile

        try:
            # Fetch all exercise outcomes
            outcomes_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("exercise_outcomes")
            )
            outcomes = list(outcomes_ref.stream())

            if not outcomes:
                profile.total_response_feedback = (
                    self._count_response_feedback(user_id)
                )
                return profile

            # Aggregate by modality and exercise
            modality_helpfulness: dict = defaultdict(list)
            modality_mood_deltas: dict = defaultdict(list)
            exercise_helpfulness: dict = defaultdict(list)

            for doc in outcomes:
                data = doc.to_dict()
                modality = data.get("modality", "unknown")
                exercise_id = data.get("exercise_id", "unknown")
                helpfulness = data.get("helpfulness", 3)
                mood_before = data.get("mood_before", 5)
                mood_after = data.get("mood_after", 5)

                modality_helpfulness[modality].append(helpfulness)
                modality_mood_deltas[modality].append(
                    mood_after - mood_before
                )
                exercise_helpfulness[exercise_id].append(helpfulness)

            profile.total_outcomes = len(outcomes)

            # Compute averages
            profile.modality_scores = {
                m: round(sum(scores) / len(scores), 1)
                for m, scores in modality_helpfulness.items()
            }
            profile.modality_mood_deltas = {
                m: round(sum(deltas) / len(deltas), 1)
                for m, deltas in modality_mood_deltas.items()
            }
            profile.exercise_scores = {
                e: round(sum(scores) / len(scores), 1)
                for e, scores in exercise_helpfulness.items()
            }

            # Favorite modality (highest average helpfulness)
            if profile.modality_scores:
                profile.favorite_modality = max(
                    profile.modality_scores,
                    key=profile.modality_scores.get,
                )

            # Response feedback stats
            feedback_data = self._get_response_feedback_stats(user_id)
            profile.total_response_feedback = feedback_data["total"]
            profile.response_helpful_rate = feedback_data["helpful_rate"]

            return profile

        except Exception as e:
            logger.warning(
                f"Failed to compute effectiveness for {user_id}: {e}"
            )
            return profile

    def _count_response_feedback(self, user_id: str) -> int:
        """Count total response feedback entries."""
        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("response_feedback")
            )
            return len(list(col_ref.stream()))
        except Exception:
            return 0

    def _get_response_feedback_stats(self, user_id: str) -> dict:
        """Get total count and helpful rate for response feedback."""
        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("response_feedback")
            )
            docs = list(col_ref.stream())
            total = len(docs)
            if total == 0:
                return {"total": 0, "helpful_rate": 0.0}

            helpful_count = sum(
                1
                for d in docs
                if d.to_dict().get("rating") == FeedbackRating.HELPFUL.value
            )
            return {
                "total": total,
                "helpful_rate": round(helpful_count / total, 2),
            }
        except Exception:
            return {"total": 0, "helpful_rate": 0.0}

    # ── Prompt Formatting ────────────────────────────────

    def format_insights_for_prompt(
        self, profile: EffectivenessProfile
    ) -> str:
        """
        Format effectiveness insights as text for system prompt injection.
        Only called when profile.total_outcomes >= 2.
        """
        parts = []

        # Favorite modality
        if profile.favorite_modality:
            modality_name = profile.favorite_modality.upper()
            score = profile.modality_scores.get(
                profile.favorite_modality, 0
            )
            parts.append(
                f"User has found {modality_name} exercises most helpful "
                f"(avg helpfulness: {score}/5)."
            )

        # Per-modality mood deltas (only meaningful ones)
        for modality, delta in profile.modality_mood_deltas.items():
            if abs(delta) >= 0.5:
                direction = "improved" if delta > 0 else "worsened"
                parts.append(
                    f"{modality.upper()} exercises {direction} mood by "
                    f"{abs(delta):.1f} points on average."
                )

        # Top individual exercises
        if profile.exercise_scores:
            top_exercises = sorted(
                profile.exercise_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            for ex_id, score in top_exercises:
                template = EXERCISE_REGISTRY.get(ex_id)
                title = template.title if template else ex_id
                parts.append(f"'{title}' rated {score}/5 helpfulness.")

        # Low-performing modalities
        for modality, score in profile.modality_scores.items():
            if score < 2.5:
                parts.append(
                    f"User has found {modality.upper()} less helpful "
                    f"(avg: {score}/5) - consider other approaches."
                )

        if not parts:
            return ""

        return (
            "--- EFFECTIVENESS INSIGHTS ---\n"
            "Based on this user's feedback history:\n"
            + "\n".join(f"  - {p}" for p in parts)
            + "\nUse these insights to tailor your approach. "
            "Lean toward techniques from modalities the user finds helpful.\n"
            "--- END INSIGHTS ---"
        )


# ── Singleton ─────────────────────────────────────────

_feedback_service: Optional[FeedbackService] = None


def get_feedback_service() -> FeedbackService:
    """Get or create FeedbackService singleton."""
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackService()
    return _feedback_service
