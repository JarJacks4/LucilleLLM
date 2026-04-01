"""
LucilleLLM - Self-Care Score Service (Phase 22)

Computes a composite engagement score (0-100) from mood history,
exercise completion, exercise effectiveness, task completion, and
consistency/streaks. NOT a clinical assessment — measures engagement
with self-care activities.

All data is read from existing services (no new Firestore collections).
Follows the singleton pattern from other services.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from cache import get_cache
from config import get_config
from models import SelfCareScore, SelfCareScoreCategory

logger = logging.getLogger(__name__)

# Default category weights (must sum to 1.0)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "Mood Stability": 0.25,
    "Exercise Engagement": 0.20,
    "Exercise Effectiveness": 0.20,
    "Task Completion": 0.15,
    "Consistency": 0.20,
}

# Minimum data thresholds per category
MIN_MOOD_ENTRIES = 3
MIN_EXERCISES_STARTED = 1
MIN_EXERCISE_OUTCOMES = 1
MIN_TASKS_ASSIGNED = 1
MIN_ACTIVE_DATES = 3

# Cache TTL for selfcare scores (seconds)
SELFCARE_CACHE_TTL = 120


def _selfcare_cache_key(user_id: str) -> str:
    return f"selfcare_score:{user_id}"


class SelfCareScoreService:
    """Computes composite self-care engagement score from existing data."""

    def __init__(self):
        self._config = get_config()
        self._cache = get_cache()

    def compute_score(self, user_id: str) -> SelfCareScore:
        """
        Compute the full Self-Care Score for a user.

        Uses TTL cache (120s) to avoid repeated Firestore queries.
        Aggregates data from UserService (mood), ProgressService (exercises,
        tasks, streaks), and FeedbackService (effectiveness). Redistributes
        weights when a category has insufficient data.
        """
        # Check cache first
        cache_key = _selfcare_cache_key(user_id)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        from user_service import get_user_service
        from progress_service import get_progress_service
        from feedback_service import get_feedback_service

        score = SelfCareScore(user_id=user_id)

        # Gather data from existing services
        user_svc = get_user_service()
        profile = user_svc.get_user_profile(user_id)

        progress_svc = get_progress_service()
        progress = progress_svc.get_progress_summary(user_id)

        feedback_svc = get_feedback_service()
        effectiveness = feedback_svc.compute_effectiveness(user_id)

        # Compute raw scores per category with sufficiency flags
        raw_scores: Dict[str, Tuple[float, str, bool]] = {}  # name -> (score, detail, has_data)

        # A. Mood Stability (25%)
        mood_score, mood_detail, mood_ok = self._compute_mood_stability(profile)
        raw_scores["Mood Stability"] = (mood_score, mood_detail, mood_ok)

        # B. Exercise Engagement (20%)
        eng_score, eng_detail, eng_ok = self._compute_exercise_engagement(progress)
        raw_scores["Exercise Engagement"] = (eng_score, eng_detail, eng_ok)

        # C. Exercise Effectiveness (20%)
        eff_score, eff_detail, eff_ok = self._compute_exercise_effectiveness(effectiveness)
        raw_scores["Exercise Effectiveness"] = (eff_score, eff_detail, eff_ok)

        # D. Task Completion (15%)
        task_score, task_detail, task_ok = self._compute_task_completion(progress)
        raw_scores["Task Completion"] = (task_score, task_detail, task_ok)

        # E. Consistency (20%)
        cons_score, cons_detail, cons_ok = self._compute_consistency(progress)
        raw_scores["Consistency"] = (cons_score, cons_detail, cons_ok)

        # Redistribute weights for categories with insufficient data
        weights = self._redistribute_weights(raw_scores)

        # Build category list and compute overall
        categories: List[SelfCareScoreCategory] = []
        overall = 0.0
        sufficient_count = sum(1 for _, _, ok in raw_scores.values() if ok)

        for name, (raw, detail, has_data) in raw_scores.items():
            w = weights[name]
            weighted = round(raw * w, 2)
            overall += weighted
            categories.append(SelfCareScoreCategory(
                name=name,
                score=round(raw, 1),
                weight=round(w, 3),
                weighted_score=round(weighted, 1),
                detail=detail if has_data else f"Insufficient data — {detail}",
            ))

        score.overall_score = round(min(overall, 100.0), 1)
        score.categories = categories

        # Data sufficiency
        if sufficient_count == 5:
            score.data_sufficiency = "full"
        elif sufficient_count >= 2:
            score.data_sufficiency = "partial"
        else:
            score.data_sufficiency = "insufficient"

        # Burnout detection
        burnout, burnout_detail = self._detect_burnout(eng_score, eff_score)
        score.burnout_flag = burnout
        score.burnout_detail = burnout_detail

        # Compute trend from available signals
        score.trend = self._compute_trend(progress, effectiveness)

        # Generate insight text
        score.insight_text = self._generate_insight(score)

        # Cache the result
        self._cache.set(cache_key, score, ttl=SELFCARE_CACHE_TTL)

        return score

    # ── Category Computations ────────────────────────────────

    def _compute_mood_stability(
        self, profile: Optional[dict]
    ) -> Tuple[float, str, bool]:
        """
        Mood Stability score from mood_history.

        Uses mean intensity (inverted: lower intensity = better stability)
        plus a variance-based stability bonus. Floor at 20.
        """
        if not profile:
            return 0.0, "No profile data available", False

        affective = profile.get("affective", {})
        mood_history = affective.get("mood_history", [])

        if not mood_history:
            return 0.0, "No mood entries recorded yet", False

        # Filter to configured window
        window_days = self._config.SELFCARE_MOOD_WINDOW_DAYS
        cutoff = datetime.now() - timedelta(days=window_days)
        recent: List[int] = []
        for entry in mood_history:
            recorded = entry.get("recorded_at", "")
            if recorded:
                try:
                    dt = datetime.fromisoformat(recorded)
                    if dt >= cutoff:
                        intensity = entry.get("intensity", 5)
                        recent.append(intensity)
                except (ValueError, TypeError):
                    pass
            else:
                # Include entries without timestamps (legacy data)
                recent.append(entry.get("intensity", 5))

        if len(recent) < MIN_MOOD_ENTRIES:
            return 0.0, f"Need at least {MIN_MOOD_ENTRIES} mood entries in the last {window_days} days", False

        # Mean mood intensity (1-10 scale)
        # Low intensity (1-4) = calm/neutral → good (score 85-100)
        # Moderate intensity (5-6) = noticeable → okay (score 65-80)
        # High intensity (7-10) = distressed → lower score (score 30-60)
        mean_intensity = statistics.mean(recent)

        # Base score: lower intensity = better stability
        # Map: intensity 1→100, 3→90, 5→75, 7→55, 10→30
        base_score = max(100 - (mean_intensity - 1) * 7.78, 30)

        # Stability bonus: low variance = good
        if len(recent) >= 2:
            variance = statistics.variance(recent)
            # Variance 0 = +15, variance 5+ = 0
            stability_bonus = max(0, 15 - variance * 3)
        else:
            stability_bonus = 7.5  # neutral if only one entry

        raw_score = min(base_score + stability_bonus, 100.0)
        # Floor at 20 to avoid penalizing honest reporters
        raw_score = max(raw_score, 20.0)

        detail = f"Based on {len(recent)} mood entries (avg intensity: {mean_intensity:.1f})"
        return raw_score, detail, True

    def _compute_exercise_engagement(self, progress) -> Tuple[float, str, bool]:
        """Exercise Engagement from completion rate + modality variety."""
        if progress.total_exercises_started < MIN_EXERCISES_STARTED:
            return 0.0, "No exercises started yet", False

        # Base: completion rate (0-1 → 0-80)
        base = progress.completion_rate * 80

        # Variety bonus: up to 20 points for using multiple modalities
        modality_count = len(progress.modality_counts)
        variety_bonus = min(modality_count * 5, 20)

        raw_score = min(base + variety_bonus, 100.0)
        detail = (
            f"{progress.total_exercises_completed}/{progress.total_exercises_started} "
            f"completed ({progress.completion_rate:.0%}), "
            f"{modality_count} modalities used"
        )
        return raw_score, detail, True

    def _compute_exercise_effectiveness(self, effectiveness) -> Tuple[float, str, bool]:
        """Exercise Effectiveness from helpfulness ratings + mood deltas."""
        if effectiveness.total_outcomes < MIN_EXERCISE_OUTCOMES:
            return 0.0, "No exercise feedback submitted yet", False

        # Helpfulness (1-5 → 0-60)
        if effectiveness.modality_scores:
            avg_helpfulness = statistics.mean(effectiveness.modality_scores.values())
        else:
            avg_helpfulness = 0
        helpfulness_score = (avg_helpfulness / 5.0) * 60

        # Mood improvement bonus (0-40)
        mood_bonus = 0.0
        if effectiveness.modality_mood_deltas:
            avg_delta = statistics.mean(effectiveness.modality_mood_deltas.values())
            # Positive delta = mood improved. Cap at 4 points improvement
            mood_bonus = min(max(avg_delta, 0) * 10, 40)

        raw_score = min(helpfulness_score + mood_bonus, 100.0)
        detail = (
            f"Avg helpfulness: {avg_helpfulness:.1f}/5, "
            f"{effectiveness.total_outcomes} outcomes recorded"
        )
        return raw_score, detail, True

    def _compute_task_completion(self, progress) -> Tuple[float, str, bool]:
        """Task Completion from task completion rate."""
        if progress.total_tasks_assigned < MIN_TASKS_ASSIGNED:
            return 0.0, "No practice tasks assigned yet", False

        raw_score = progress.task_completion_rate * 100.0
        detail = (
            f"{progress.total_tasks_completed}/{progress.total_tasks_assigned} "
            f"tasks completed ({progress.task_completion_rate:.0%})"
        )
        return raw_score, detail, True

    def _compute_consistency(self, progress) -> Tuple[float, str, bool]:
        """Consistency from streak days + weekly session frequency."""
        has_data = (
            progress.current_streak_days > 0
            or progress.total_exercises_completed >= MIN_ACTIVE_DATES
            or progress.sessions_this_week > 0
        )
        if not has_data:
            return 0.0, "Not enough activity to measure consistency", False

        cap = self._config.SELFCARE_STREAK_CAP_DAYS

        # Streak contribution (0-50): current streak / cap
        streak_score = min(progress.current_streak_days / cap, 1.0) * 50

        # Weekly session frequency contribution (0-50): 5+ sessions/week = full score
        weekly_score = min(progress.sessions_this_week / 5, 1.0) * 50

        raw_score = min(streak_score + weekly_score, 100.0)
        detail = (
            f"{progress.current_streak_days}-day streak, "
            f"{progress.sessions_this_week} sessions this week"
        )
        return raw_score, detail, True

    # ── Weight Redistribution ────────────────────────────────

    def _redistribute_weights(
        self, raw_scores: Dict[str, Tuple[float, str, bool]]
    ) -> Dict[str, float]:
        """
        Redistribute weights from insufficient-data categories
        to categories that have enough data. Returns adjusted weights.
        """
        weights = dict(DEFAULT_WEIGHTS)
        insufficient = [name for name, (_, _, ok) in raw_scores.items() if not ok]
        sufficient = [name for name, (_, _, ok) in raw_scores.items() if ok]

        if not insufficient:
            return weights
        if not sufficient:
            # All categories lack data — zero everything
            return {name: 0.0 for name in weights}

        # Sum weight from insufficient categories
        freed_weight = sum(weights[name] for name in insufficient)
        # Set insufficient categories to 0
        for name in insufficient:
            weights[name] = 0.0
        # Distribute freed weight proportionally to sufficient categories
        total_sufficient = sum(weights[name] for name in sufficient)
        if total_sufficient > 0:
            for name in sufficient:
                weights[name] += freed_weight * (weights[name] / total_sufficient)

        return weights

    # ── Burnout Detection ────────────────────────────────────

    def _detect_burnout(
        self, engagement_score: float, effectiveness_score: float
    ) -> Tuple[bool, str]:
        """
        Detect burnout: high engagement but low effectiveness.
        """
        eng_threshold = self._config.SELFCARE_BURNOUT_ENGAGEMENT_THRESHOLD
        eff_threshold = self._config.SELFCARE_BURNOUT_EFFECTIVENESS_THRESHOLD

        if engagement_score >= eng_threshold and effectiveness_score <= eff_threshold:
            return True, (
                "You've been very active, but the exercises may not be landing as well. "
                "Consider trying a different approach or taking a mindful pause."
            )
        return False, ""

    # ── Trend Computation ─────────────────────────────────────

    def _compute_trend(self, progress, effectiveness) -> str:
        """
        Estimate trend from available signals.

        Uses: current streak direction, completion rate, and effectiveness.
        Returns 'improving', 'declining', or 'stable'.
        """
        signals = 0  # positive = improving, negative = declining

        # Signal 1: Active streak suggests improving
        if progress.current_streak_days >= 3:
            signals += 1
        elif progress.current_streak_days == 0 and progress.total_exercises_completed > 0:
            signals -= 1  # Had activity before but broke streak

        # Signal 2: High completion rate suggests improving
        if progress.completion_rate >= 0.7:
            signals += 1
        elif progress.completion_rate < 0.3 and progress.total_exercises_started >= 3:
            signals -= 1

        # Signal 3: Good effectiveness suggests improving
        if effectiveness.modality_scores:
            avg = statistics.mean(effectiveness.modality_scores.values())
            if avg >= 4.0:
                signals += 1
            elif avg < 2.5:
                signals -= 1

        # Signal 4: Weekly session frequency
        if progress.sessions_this_week >= 3:
            signals += 1
        elif progress.sessions_this_week == 0 and progress.total_exercises_completed > 0:
            signals -= 1

        if signals >= 2:
            return "improving"
        elif signals <= -2:
            return "declining"
        return "stable"

    # ── Insight Generation ───────────────────────────────────

    def _generate_insight(self, score: SelfCareScore) -> str:
        """Generate a template-based insight text (no LLM call)."""
        if score.data_sufficiency == "insufficient":
            return (
                "Keep going! Once you have a few more mood entries and completed "
                "exercises, Lucille can give you a personalized self-care snapshot."
            )

        if score.burnout_flag:
            return score.burnout_detail

        # Find the lowest-scoring category with data
        scored_cats = [c for c in score.categories if c.weight > 0]
        if not scored_cats:
            return ""

        lowest = min(scored_cats, key=lambda c: c.score)
        overall = score.overall_score

        if overall >= 80:
            return (
                f"Excellent self-care engagement! Your overall score is {overall:.0f}. "
                f"Keep up the great work."
            )
        elif overall >= 60:
            return (
                f"You're doing well with a score of {overall:.0f}. "
                f"Your '{lowest.name}' area could use a bit more attention."
            )
        elif overall >= 40:
            return (
                f"Your self-care score is {overall:.0f} — there's room to grow. "
                f"Try focusing on '{lowest.name}' this week."
            )
        else:
            return (
                f"Your score is {overall:.0f}. Small steps make a big difference — "
                f"even one exercise or mood check-in today helps build momentum."
            )


# ── Singleton ─────────────────────────────────────────────

_selfcare_score_service: Optional[SelfCareScoreService] = None


def get_selfcare_score_service() -> SelfCareScoreService:
    """Get or create SelfCareScoreService singleton."""
    global _selfcare_score_service
    if _selfcare_score_service is None:
        _selfcare_score_service = SelfCareScoreService()
    return _selfcare_score_service
