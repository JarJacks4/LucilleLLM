"""
LucilleLLM - RL-Based Intervention Selection Service

Thompson Sampling (Beta distributions) for contextual bandit-based
exercise recommendation. Uses emotion groups as context and
(emotion_group, exercise_id) pairs as arms.

No external ML dependencies: uses Python's built-in random.betavariate.

Firestore structure: bandit_state/{user_id} (single document per user)

Follows the singleton pattern from other services.
"""

import logging
import random
from datetime import datetime
from typing import Dict, List, Optional

from config import get_config

logger = logging.getLogger(__name__)


# ── Emotion Group Mapping ─────────────────────────────
# Maps raw detected emotions to 5 groups to reduce arm count.
# All DetectedEmotion enum values are covered.

EMOTION_GROUPS: Dict[str, str] = {
    # anxious group
    "anxious": "anxious",
    "worried": "anxious",
    "stressed": "anxious",
    "overwhelmed": "anxious",
    "fearful": "anxious",
    # sad group
    "sad": "sad",
    "hopeless": "sad",
    "lonely": "sad",
    "grieving": "sad",
    # angry group
    "angry": "angry",
    "frustrated": "angry",
    "irritable": "angry",
    "disgusted": "angry",
    # neutral group
    "neutral": "neutral",
    "calm": "neutral",
    "curious": "neutral",
    "surprised": "neutral",
    # positive group
    "happy": "positive",
    "grateful": "positive",
    "hopeful": "positive",
}


def _arm_key(emotion_group: str, exercise_id: str) -> str:
    """Create a string key for a bandit arm."""
    return f"({emotion_group},{exercise_id})"


class RLService:
    """
    Thompson Sampling service for exercise recommendation.

    Each (emotion_group, exercise_id) pair is a bandit arm with
    Beta(alpha, beta) distribution parameters.

    Arms are stored per-user in Firestore:
        bandit_state/{user_id} -> { arms: {arm_key: {alpha, beta}, ...} }
    """

    COLLECTION = "bandit_state"

    def __init__(self):
        self._config = get_config()
        self._db = None

        try:
            from firebase_service import get_firebase_service
            fb = get_firebase_service()
            self._db = fb.db
        except Exception as e:
            logger.warning(f"RLService: Firebase not available — {e}")

        logger.info(
            f"RLService initialized — "
            f"rl_enabled={self._config.RL_ENABLED}, "
            f"exploration_bonus={self._config.RL_EXPLORATION_BONUS}, "
            f"success_threshold={self._config.RL_SUCCESS_THRESHOLD}, "
            f"db_available={self._db is not None}"
        )

    # ── Properties ────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Check if RL service has database access and is enabled."""
        return self._db is not None and self._config.RL_ENABLED

    # ── Emotion Grouping ──────────────────────────────────

    def get_emotion_group(self, emotion: str) -> str:
        """Map a raw emotion string to its emotion group."""
        return EMOTION_GROUPS.get(emotion.lower(), "neutral")

    # ── Bandit State Persistence ──────────────────────────

    def load_bandit_state(self, user_id: str) -> Dict[str, dict]:
        """
        Load the bandit arm states for a user from Firestore.

        Returns a dict of arm_key -> {"alpha": float, "beta": float}.
        Returns empty dict if no state exists or Firestore unavailable.
        """
        if self._db is None:
            return {}

        try:
            doc = (
                self._db.collection(self.COLLECTION)
                .document(user_id)
                .get()
            )
            if doc.exists:
                data = doc.to_dict()
                return data.get("arms", {})
            return {}
        except Exception as e:
            logger.warning(f"Failed to load bandit state for {user_id}: {e}")
            return {}

    def save_bandit_state(
        self, user_id: str, arms: Dict[str, dict]
    ) -> bool:
        """
        Save the bandit arm states for a user to Firestore.
        Uses set with merge to avoid overwriting other fields.
        """
        if self._db is None:
            return False

        try:
            doc_ref = (
                self._db.collection(self.COLLECTION)
                .document(user_id)
            )
            doc_ref.set(
                {
                    "arms": arms,
                    "updated_at": datetime.now().isoformat(),
                },
                merge=True,
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to save bandit state for {user_id}: {e}")
            return False

    # ── Thompson Sampling Core ────────────────────────────

    def sample_thompson_scores(
        self,
        user_id: str,
        emotion: str,
        exercise_ids: List[str],
    ) -> Dict[str, float]:
        """
        Sample from Beta distributions for each eligible exercise
        in the given emotion context.

        Returns a dict of exercise_id -> sampled_score (0.0 to 1.0).
        Exercises without history get uniform prior (alpha=1, beta=1).
        """
        emotion_group = self.get_emotion_group(emotion)
        arms = self.load_bandit_state(user_id)

        scores: Dict[str, float] = {}
        for exercise_id in exercise_ids:
            key = _arm_key(emotion_group, exercise_id)
            arm = arms.get(key, {"alpha": 1.0, "beta": 1.0})
            alpha = max(arm.get("alpha", 1.0), 0.01)
            beta_val = max(arm.get("beta", 1.0), 0.01)

            # Sample from Beta distribution
            sampled = random.betavariate(alpha, beta_val)
            scores[exercise_id] = sampled

        return scores

    # ── Reward Update ─────────────────────────────────────

    def update_reward(
        self,
        user_id: str,
        exercise_id: str,
        emotion: str,
        helpfulness: int,
        mood_before: int,
        mood_after: int,
        would_repeat: bool,
    ) -> bool:
        """
        Update the bandit arm based on exercise outcome.

        Primary signal: helpfulness >= RL_SUCCESS_THRESHOLD -> success (alpha += 1)
                        helpfulness <  RL_SUCCESS_THRESHOLD -> failure (beta += 1)

        Secondary signals (fractional adjustments for nuanced learning):
          - Positive mood_delta (>= 2): alpha += 0.5
          - Negative mood_delta (<= -2): beta += 0.5
          - would_repeat = True: alpha += 0.25
          - would_repeat = False: beta += 0.25

        Returns True if the update was saved successfully.
        """
        emotion_group = self.get_emotion_group(emotion)
        arms = self.load_bandit_state(user_id)
        key = _arm_key(emotion_group, exercise_id)

        arm = arms.get(key, {"alpha": 1.0, "beta": 1.0})
        alpha = arm.get("alpha", 1.0)
        beta_val = arm.get("beta", 1.0)

        success_threshold = self._config.RL_SUCCESS_THRESHOLD

        # Primary signal
        if helpfulness >= success_threshold:
            alpha += 1.0
        else:
            beta_val += 1.0

        # Secondary: mood delta
        mood_delta = mood_after - mood_before
        if mood_delta >= 2:
            alpha += 0.5
        elif mood_delta <= -2:
            beta_val += 0.5

        # Secondary: would_repeat
        if would_repeat:
            alpha += 0.25
        else:
            beta_val += 0.25

        arms[key] = {"alpha": round(alpha, 2), "beta": round(beta_val, 2)}

        logger.info(
            f"RL reward update: user={user_id}, arm={key}, "
            f"helpfulness={helpfulness}, alpha={alpha:.2f}, beta={beta_val:.2f}"
        )

        return self.save_bandit_state(user_id, arms)

    # ── Diagnostics ───────────────────────────────────────

    def get_arm_stats(
        self, user_id: str, emotion: str
    ) -> List[dict]:
        """
        Get human-readable arm statistics for a given emotion context.
        Useful for debugging and the diagnostic endpoint.
        """
        emotion_group = self.get_emotion_group(emotion)
        arms = self.load_bandit_state(user_id)

        stats = []
        for key, arm in arms.items():
            if key.startswith(f"({emotion_group},"):
                # Extract exercise_id from key "(emotion_group,exercise_id)"
                exercise_id = key.split(",", 1)[1].rstrip(")")
                alpha = arm.get("alpha", 1.0)
                beta_val = arm.get("beta", 1.0)
                total = alpha + beta_val - 2  # subtract priors
                mean = alpha / (alpha + beta_val) if (alpha + beta_val) > 0 else 0.5
                stats.append({
                    "exercise_id": exercise_id,
                    "emotion_group": emotion_group,
                    "alpha": alpha,
                    "beta": beta_val,
                    "mean_success_rate": round(mean, 3),
                    "total_observations": round(total, 1),
                })

        # Sort by mean success rate descending
        stats.sort(key=lambda x: x["mean_success_rate"], reverse=True)
        return stats

    def get_all_arm_stats(self, user_id: str) -> List[dict]:
        """Get all arm statistics across all emotion groups."""
        arms = self.load_bandit_state(user_id)

        stats = []
        for key, arm in arms.items():
            # Parse key "(emotion_group,exercise_id)"
            inner = key.strip("()")
            parts = inner.split(",", 1)
            if len(parts) != 2:
                continue
            emotion_group, exercise_id = parts

            alpha = arm.get("alpha", 1.0)
            beta_val = arm.get("beta", 1.0)
            total = alpha + beta_val - 2
            mean = alpha / (alpha + beta_val) if (alpha + beta_val) > 0 else 0.5
            stats.append({
                "exercise_id": exercise_id,
                "emotion_group": emotion_group,
                "alpha": alpha,
                "beta": beta_val,
                "mean_success_rate": round(mean, 3),
                "total_observations": round(total, 1),
            })

        stats.sort(key=lambda x: x["mean_success_rate"], reverse=True)
        return stats


# ── Singleton ─────────────────────────────────────────

_rl_service: Optional[RLService] = None


def get_rl_service() -> RLService:
    """Get or create RLService singleton."""
    global _rl_service
    if _rl_service is None:
        _rl_service = RLService()
    return _rl_service
