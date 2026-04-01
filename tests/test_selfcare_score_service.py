"""
Tests for SelfCareScoreService — composite engagement scoring.

Verifies score computation, weight redistribution, burnout detection,
mood stability floor, and data sufficiency reporting.
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selfcare_score_service import (
    SelfCareScoreService,
    DEFAULT_WEIGHTS,
    MIN_MOOD_ENTRIES,
)
from models import SelfCareScore, SelfCareScoreCategory, ProgressSummary, EffectivenessProfile


@pytest.fixture(autouse=True)
def fresh_service():
    """Reset the singleton and cache before each test."""
    import selfcare_score_service
    selfcare_score_service._selfcare_score_service = None
    # Clear cache entries for selfcare scores
    from cache import get_cache
    get_cache().invalidate_prefix("selfcare_score:")
    yield


def _make_mood_history(entries, days_back=14):
    """Helper: create mood_history entries within the configured window."""
    now = datetime.now()
    result = []
    for i, intensity in enumerate(entries):
        ts = (now - timedelta(days=days_back - i)).isoformat()
        result.append({
            "mood": "neutral",
            "intensity": intensity,
            "context": "test",
            "recorded_at": ts,
            "detected_via": "manual",
            "confidence": 1.0,
        })
    return result


def _make_progress(
    started=5,
    completed=4,
    abandoned=1,
    modality_counts=None,
    streak=3,
    longest_streak=5,
    tasks_assigned=3,
    tasks_completed=2,
    sessions_this_week=3,
):
    """Helper: create a ProgressSummary with test data."""
    p = ProgressSummary(user_id="test_user")
    p.total_exercises_started = started
    p.total_exercises_completed = completed
    p.total_exercises_abandoned = abandoned
    p.completion_rate = round(completed / started, 2) if started > 0 else 0.0
    p.modality_counts = modality_counts or {"cbt": 2, "dbt": 2}
    p.current_streak_days = streak
    p.longest_streak_days = longest_streak
    p.total_tasks_assigned = tasks_assigned
    p.total_tasks_completed = tasks_completed
    p.task_completion_rate = round(tasks_completed / tasks_assigned, 2) if tasks_assigned > 0 else 0.0
    p.total_practice_minutes = completed * 15
    p.sessions_this_week = sessions_this_week
    return p


def _make_effectiveness(
    modality_scores=None,
    modality_mood_deltas=None,
    total_outcomes=3,
):
    """Helper: create an EffectivenessProfile with test data."""
    e = EffectivenessProfile(user_id="test_user")
    e.modality_scores = modality_scores or {"cbt": 4.0, "dbt": 3.5}
    e.modality_mood_deltas = modality_mood_deltas or {"cbt": 2.0, "dbt": 1.0}
    e.total_outcomes = total_outcomes
    e.favorite_modality = "cbt"
    return e


class TestSelfCareScoreFullData:
    """Tests with complete data across all categories."""

    def test_full_data_returns_all_five_categories(self):
        svc = SelfCareScoreService()

        profile = {
            "affective": {"mood_history": _make_mood_history([4, 5, 4, 5, 4])}
        }
        progress = _make_progress()
        effectiveness = _make_effectiveness()

        with patch("user_service.get_user_service") as mock_user, \
             patch("progress_service.get_progress_service") as mock_prog, \
             patch("feedback_service.get_feedback_service") as mock_fb:

            mock_user().get_user_profile.return_value = profile
            mock_prog().get_progress_summary.return_value = progress
            mock_fb().compute_effectiveness.return_value = effectiveness

            result = svc.compute_score("test_user")

        assert len(result.categories) == 5
        assert result.data_sufficiency == "full"
        assert result.overall_score > 0
        assert result.overall_score <= 100

    def test_category_weights_sum_to_one(self):
        svc = SelfCareScoreService()

        profile = {
            "affective": {"mood_history": _make_mood_history([4, 5, 4, 5, 4])}
        }
        progress = _make_progress()
        effectiveness = _make_effectiveness()

        with patch("user_service.get_user_service") as mock_user, \
             patch("progress_service.get_progress_service") as mock_prog, \
             patch("feedback_service.get_feedback_service") as mock_fb:

            mock_user().get_user_profile.return_value = profile
            mock_prog().get_progress_summary.return_value = progress
            mock_fb().compute_effectiveness.return_value = effectiveness

            result = svc.compute_score("test_user")

        total_weight = sum(c.weight for c in result.categories)
        assert abs(total_weight - 1.0) < 0.01

    def test_overall_equals_sum_of_weighted_scores(self):
        svc = SelfCareScoreService()

        profile = {
            "affective": {"mood_history": _make_mood_history([4, 5, 4, 5, 4])}
        }
        progress = _make_progress()
        effectiveness = _make_effectiveness()

        with patch("user_service.get_user_service") as mock_user, \
             patch("progress_service.get_progress_service") as mock_prog, \
             patch("feedback_service.get_feedback_service") as mock_fb:

            mock_user().get_user_profile.return_value = profile
            mock_prog().get_progress_summary.return_value = progress
            mock_fb().compute_effectiveness.return_value = effectiveness

            result = svc.compute_score("test_user")

        expected = sum(c.weighted_score for c in result.categories)
        assert abs(result.overall_score - round(expected, 1)) < 0.2


class TestSelfCareScoreNoData:
    """Tests with no data at all."""

    def test_no_data_returns_zero_score(self):
        svc = SelfCareScoreService()

        progress = ProgressSummary(user_id="test_user")
        effectiveness = EffectivenessProfile(user_id="test_user")

        with patch("user_service.get_user_service") as mock_user, \
             patch("progress_service.get_progress_service") as mock_prog, \
             patch("feedback_service.get_feedback_service") as mock_fb:

            mock_user().get_user_profile.return_value = None
            mock_prog().get_progress_summary.return_value = progress
            mock_fb().compute_effectiveness.return_value = effectiveness

            result = svc.compute_score("test_user")

        assert result.overall_score == 0.0
        assert result.data_sufficiency == "insufficient"

    def test_no_data_has_encouragement_insight(self):
        svc = SelfCareScoreService()

        progress = ProgressSummary(user_id="test_user")
        effectiveness = EffectivenessProfile(user_id="test_user")

        with patch("user_service.get_user_service") as mock_user, \
             patch("progress_service.get_progress_service") as mock_prog, \
             patch("feedback_service.get_feedback_service") as mock_fb:

            mock_user().get_user_profile.return_value = None
            mock_prog().get_progress_summary.return_value = progress
            mock_fb().compute_effectiveness.return_value = effectiveness

            result = svc.compute_score("test_user")

        assert "Keep going" in result.insight_text


class TestSelfCareScorePartialData:
    """Tests with partial data — weight redistribution."""

    def test_partial_data_redistributes_weights(self):
        svc = SelfCareScoreService()

        # Only mood data, no exercises/tasks/feedback
        profile = {
            "affective": {"mood_history": _make_mood_history([4, 5, 4, 5, 4])}
        }
        progress = ProgressSummary(user_id="test_user")
        effectiveness = EffectivenessProfile(user_id="test_user")

        with patch("user_service.get_user_service") as mock_user, \
             patch("progress_service.get_progress_service") as mock_prog, \
             patch("feedback_service.get_feedback_service") as mock_fb:

            mock_user().get_user_profile.return_value = profile
            mock_prog().get_progress_summary.return_value = progress
            mock_fb().compute_effectiveness.return_value = effectiveness

            result = svc.compute_score("test_user")

        assert result.data_sufficiency == "insufficient"  # Only 1 category has data
        # Mood Stability should have all the weight
        mood_cat = next(c for c in result.categories if c.name == "Mood Stability")
        assert mood_cat.weight == 1.0
        # Other categories should have 0 weight
        for c in result.categories:
            if c.name != "Mood Stability":
                assert c.weight == 0.0

    def test_two_categories_partial(self):
        svc = SelfCareScoreService()

        profile = {
            "affective": {"mood_history": _make_mood_history([4, 5, 4, 5, 4])}
        }
        progress = _make_progress(tasks_assigned=0, tasks_completed=0)
        effectiveness = EffectivenessProfile(user_id="test_user")

        with patch("user_service.get_user_service") as mock_user, \
             patch("progress_service.get_progress_service") as mock_prog, \
             patch("feedback_service.get_feedback_service") as mock_fb:

            mock_user().get_user_profile.return_value = profile
            mock_prog().get_progress_summary.return_value = progress
            mock_fb().compute_effectiveness.return_value = effectiveness

            result = svc.compute_score("test_user")

        # Should be "partial" with mood + engagement + consistency having data
        assert result.data_sufficiency == "partial"
        total_weight = sum(c.weight for c in result.categories)
        assert abs(total_weight - 1.0) < 0.01


class TestMoodStability:
    """Tests for mood stability scoring logic."""

    def test_mood_floor_at_20(self):
        """Even extreme mood values should not drop below floor of 20."""
        svc = SelfCareScoreService()

        # Very high intensity + high variance
        entries = _make_mood_history([10, 1, 10, 1, 10])
        score, detail, ok = svc._compute_mood_stability(
            {"affective": {"mood_history": entries}}
        )
        assert ok is True
        assert score >= 20.0

    def test_stable_moderate_mood_scores_high(self):
        """Stable moderate intensity should score well."""
        svc = SelfCareScoreService()

        entries = _make_mood_history([4, 4, 4, 4, 4])
        score, detail, ok = svc._compute_mood_stability(
            {"affective": {"mood_history": entries}}
        )
        assert ok is True
        assert score >= 75.0

    def test_calm_low_intensity_scores_high(self):
        """Low intensity (calm/neutral) should score very well, not be penalized."""
        svc = SelfCareScoreService()

        entries = _make_mood_history([1, 2, 1, 2, 1])
        score, detail, ok = svc._compute_mood_stability(
            {"affective": {"mood_history": entries}}
        )
        assert ok is True
        assert score >= 90.0, f"Calm user (avg intensity ~1.4) should score >= 90, got {score}"

    def test_insufficient_mood_entries(self):
        """Below MIN_MOOD_ENTRIES should report insufficient."""
        svc = SelfCareScoreService()

        entries = _make_mood_history([5, 5])  # Only 2, need 3
        score, detail, ok = svc._compute_mood_stability(
            {"affective": {"mood_history": entries}}
        )
        assert ok is False
        assert score == 0.0


class TestBurnoutDetection:
    """Tests for burnout detection."""

    def test_burnout_triggers_on_high_engagement_low_effectiveness(self):
        svc = SelfCareScoreService()
        burnout, detail = svc._detect_burnout(
            engagement_score=90.0, effectiveness_score=20.0
        )
        assert burnout is True
        assert len(detail) > 0

    def test_no_burnout_when_both_high(self):
        svc = SelfCareScoreService()
        burnout, detail = svc._detect_burnout(
            engagement_score=90.0, effectiveness_score=80.0
        )
        assert burnout is False

    def test_no_burnout_when_both_low(self):
        svc = SelfCareScoreService()
        burnout, detail = svc._detect_burnout(
            engagement_score=30.0, effectiveness_score=20.0
        )
        assert burnout is False


class TestWeightRedistribution:
    """Tests for weight redistribution logic."""

    def test_all_sufficient_keeps_default_weights(self):
        svc = SelfCareScoreService()
        raw_scores = {
            name: (50.0, "detail", True) for name in DEFAULT_WEIGHTS
        }
        weights = svc._redistribute_weights(raw_scores)
        for name, w in weights.items():
            assert abs(w - DEFAULT_WEIGHTS[name]) < 0.001

    def test_one_insufficient_redistributes(self):
        svc = SelfCareScoreService()
        raw_scores = {
            name: (50.0, "detail", True) for name in DEFAULT_WEIGHTS
        }
        raw_scores["Task Completion"] = (0.0, "no data", False)
        weights = svc._redistribute_weights(raw_scores)

        assert weights["Task Completion"] == 0.0
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001

    def test_all_insufficient_keeps_zero(self):
        svc = SelfCareScoreService()
        raw_scores = {
            name: (0.0, "no data", False) for name in DEFAULT_WEIGHTS
        }
        weights = svc._redistribute_weights(raw_scores)
        for w in weights.values():
            assert w == 0.0


class TestInsightGeneration:
    """Tests for template-based insight generation."""

    def test_insufficient_data_insight(self):
        svc = SelfCareScoreService()
        score = SelfCareScore(
            user_id="test_user",
            data_sufficiency="insufficient",
        )
        insight = svc._generate_insight(score)
        assert "Keep going" in insight

    def test_high_score_insight(self):
        svc = SelfCareScoreService()
        score = SelfCareScore(
            user_id="test_user",
            overall_score=85.0,
            data_sufficiency="full",
            categories=[
                SelfCareScoreCategory(name="Mood Stability", score=90, weight=0.25, weighted_score=22.5),
                SelfCareScoreCategory(name="Consistency", score=80, weight=0.20, weighted_score=16.0),
            ],
        )
        insight = svc._generate_insight(score)
        assert "Excellent" in insight


class TestTrendComputation:
    """Tests for trend computation."""

    def test_improving_trend(self):
        svc = SelfCareScoreService()
        progress = _make_progress(
            started=10, completed=8, streak=5, sessions_this_week=4
        )
        effectiveness = _make_effectiveness(modality_scores={"cbt": 4.5})
        trend = svc._compute_trend(progress, effectiveness)
        assert trend == "improving"

    def test_declining_trend(self):
        svc = SelfCareScoreService()
        progress = _make_progress(
            started=10, completed=2, streak=0, sessions_this_week=0
        )
        effectiveness = _make_effectiveness(modality_scores={"cbt": 2.0})
        trend = svc._compute_trend(progress, effectiveness)
        assert trend == "declining"

    def test_stable_trend(self):
        svc = SelfCareScoreService()
        progress = _make_progress(
            started=3, completed=2, streak=1, sessions_this_week=1
        )
        effectiveness = _make_effectiveness(modality_scores={"cbt": 3.0})
        trend = svc._compute_trend(progress, effectiveness)
        assert trend == "stable"


class TestCaching:
    """Tests for score caching."""

    def test_cached_result_returned(self):
        svc = SelfCareScoreService()

        profile = {
            "affective": {"mood_history": _make_mood_history([4, 5, 4, 5, 4])}
        }
        progress = _make_progress()
        effectiveness = _make_effectiveness()

        with patch("user_service.get_user_service") as mock_user, \
             patch("progress_service.get_progress_service") as mock_prog, \
             patch("feedback_service.get_feedback_service") as mock_fb:

            mock_user().get_user_profile.return_value = profile
            mock_prog().get_progress_summary.return_value = progress
            mock_fb().compute_effectiveness.return_value = effectiveness

            # First call — computes
            result1 = svc.compute_score("test_user")
            # Second call — should use cache
            result2 = svc.compute_score("test_user")

        assert result1.overall_score == result2.overall_score
        assert result1.computed_at == result2.computed_at  # Same object from cache


class TestDisclaimer:
    """Ensure the disclaimer is always present."""

    def test_disclaimer_present(self):
        svc = SelfCareScoreService()

        progress = ProgressSummary(user_id="test_user")
        effectiveness = EffectivenessProfile(user_id="test_user")

        with patch("user_service.get_user_service") as mock_user, \
             patch("progress_service.get_progress_service") as mock_prog, \
             patch("feedback_service.get_feedback_service") as mock_fb:

            mock_user().get_user_profile.return_value = None
            mock_prog().get_progress_summary.return_value = progress
            mock_fb().compute_effectiveness.return_value = effectiveness

            result = svc.compute_score("test_user")

        assert "not" in result.disclaimer.lower()
        assert "clinical" in result.disclaimer.lower()
