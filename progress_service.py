"""
LucilleLLM - Progress Tracking Service

Computes analytics (completion rates, streaks, modality breakdown)
from exercise and task history. No persistent state — all computed
at query time from Firestore.

Follows the singleton pattern from other services.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, Set, Tuple

from firebase_service import get_firebase_service
from feedback_service import get_feedback_service
from therapy_service import EXERCISE_REGISTRY
from models import ProgressSummary

logger = logging.getLogger(__name__)


class ProgressService:
    """Computes progress analytics from exercise/task history."""

    COLLECTION = "exercise_sessions"  # Same collection as TherapyService

    def __init__(self):
        self._firebase = get_firebase_service()

    @property
    def db(self):
        return self._firebase.db

    def get_progress_summary(self, user_id: str) -> ProgressSummary:
        """Compute a full progress summary for a user."""
        summary = ProgressSummary(user_id=user_id)

        if self.db is None:
            return summary

        try:
            # Fetch all exercise sessions
            sessions_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("sessions")
            )
            sessions = list(sessions_ref.stream())

            completed_dates: Set[object] = set()
            modality_counts: dict = defaultdict(int)
            total_minutes = 0

            for doc in sessions:
                data = doc.to_dict()
                status = data.get("status", "")
                summary.total_exercises_started += 1

                if status == "completed":
                    summary.total_exercises_completed += 1
                    modality = data.get("modality", "unknown")
                    modality_counts[modality] += 1

                    # Track completion date for streaks
                    completed_at = data.get("completed_at", "")
                    if completed_at:
                        try:
                            dt = datetime.fromisoformat(completed_at)
                            completed_dates.add(dt.date())
                        except (ValueError, TypeError):
                            pass

                    # Estimate practice time from template
                    ex_id = data.get("exercise_id", "")
                    template = EXERCISE_REGISTRY.get(ex_id)
                    if template:
                        total_minutes += template.duration_minutes

                elif status == "abandoned":
                    summary.total_exercises_abandoned += 1

            # Sessions this week (last 7 days)
            week_ago = (datetime.now() - timedelta(days=7)).date()
            summary.sessions_this_week = sum(
                1 for d in completed_dates if d >= week_ago
            )

            # Completion rate
            if summary.total_exercises_started > 0:
                summary.completion_rate = round(
                    summary.total_exercises_completed
                    / summary.total_exercises_started,
                    2,
                )

            summary.modality_counts = dict(modality_counts)
            summary.total_practice_minutes = total_minutes

            # Fetch task data
            tasks_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("tasks")
            )
            tasks = list(tasks_ref.stream())

            task_completed_dates: Set[object] = set()
            for doc in tasks:
                data = doc.to_dict()
                summary.total_tasks_assigned += 1
                if data.get("status") == "completed":
                    summary.total_tasks_completed += 1
                    completed_at = data.get("completed_at", "")
                    if completed_at:
                        try:
                            dt = datetime.fromisoformat(completed_at)
                            task_completed_dates.add(dt.date())
                        except (ValueError, TypeError):
                            pass

            if summary.total_tasks_assigned > 0:
                summary.task_completion_rate = round(
                    summary.total_tasks_completed
                    / summary.total_tasks_assigned,
                    2,
                )

            # Compute streaks from combined dates
            all_active_dates = completed_dates | task_completed_dates
            current_streak, longest_streak = self._compute_streaks(
                all_active_dates
            )
            summary.current_streak_days = current_streak
            summary.longest_streak_days = longest_streak

            # Phase 7: Fetch feedback stats
            try:
                feedback_svc = get_feedback_service()
                effectiveness = feedback_svc.compute_effectiveness(user_id)
                summary.total_feedback_given = (
                    effectiveness.total_response_feedback
                )
                summary.average_helpfulness = (
                    round(
                        sum(effectiveness.modality_scores.values())
                        / len(effectiveness.modality_scores),
                        1,
                    )
                    if effectiveness.modality_scores
                    else 0.0
                )
                summary.favorite_modality = effectiveness.favorite_modality
            except Exception as e:
                logger.warning(f"Failed to compute feedback stats: {e}")

            return summary

        except Exception as e:
            logger.warning(f"Failed to compute progress for {user_id}: {e}")
            return summary

    def _compute_streaks(self, active_dates: set) -> Tuple[int, int]:
        """
        Compute current and longest streak from a set of dates.
        Returns (current_streak, longest_streak).
        """
        if not active_dates:
            return 0, 0

        sorted_dates = sorted(active_dates)
        today = datetime.now().date()

        # Longest streak
        longest = 1
        current_run = 1
        for i in range(1, len(sorted_dates)):
            if (sorted_dates[i] - sorted_dates[i - 1]).days == 1:
                current_run += 1
                longest = max(longest, current_run)
            else:
                current_run = 1

        # Current streak (must include today or yesterday)
        current = 0
        check_date = today
        # Allow yesterday to count as "still on streak"
        if check_date not in active_dates:
            check_date = today - timedelta(days=1)

        while check_date in active_dates:
            current += 1
            check_date -= timedelta(days=1)

        return current, longest


# ── Singleton ─────────────────────────────────────────

_progress_service: Optional[ProgressService] = None


def get_progress_service() -> ProgressService:
    """Get or create ProgressService singleton."""
    global _progress_service
    if _progress_service is None:
        _progress_service = ProgressService()
    return _progress_service
