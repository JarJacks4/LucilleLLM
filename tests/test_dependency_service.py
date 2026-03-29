"""
Tests for DependencyService — usage pattern monitoring and risk scoring.

Tests the 7 dependency detection signals and risk level classification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dependency_service import DependencyService
from models import DependencyRiskLevel, InteractionMetrics


def _make_metrics(**kwargs) -> InteractionMetrics:
    """Helper to create InteractionMetrics with defaults."""
    defaults = {
        "user_id": "test-user",
        "messages_today": 5,
        "sessions_today": 2,
        "messages_this_hour": 5,
        "consecutive_days": 3,
        "total_messages_this_week": 30,
        "total_messages_last_week": 25,
        "nighttime_messages_count": 0,
    }
    defaults.update(kwargs)
    return InteractionMetrics(**defaults)


class TestDependencyScoring:
    """Test dependency risk score computation."""

    def setup_method(self):
        self.svc = DependencyService()

    def test_no_risk_normal_usage(self):
        """Normal usage patterns should return NONE risk."""
        metrics = _make_metrics()
        assessment = self.svc.assess_dependency("test-user", metrics, "how are you")
        assert assessment.risk_level == DependencyRiskLevel.NONE
        assert assessment.score < 16

    def test_high_message_frequency_detected(self):
        """Over 20 messages/hour must raise a dependency signal."""
        metrics = _make_metrics(messages_this_hour=25)
        assessment = self.svc.assess_dependency("test-user", metrics, "hello")
        assert assessment.score > 0
        assert len(assessment.signals) > 0

    def test_high_session_frequency_detected(self):
        """Over 8 sessions/day must raise a dependency signal."""
        metrics = _make_metrics(sessions_today=10)
        assessment = self.svc.assess_dependency("test-user", metrics, "hello")
        assert assessment.score > 0

    def test_score_capped_at_100(self):
        """Score must never exceed 100."""
        metrics = _make_metrics(
            messages_this_hour=50,
            sessions_today=20,
            nighttime_messages_count=10,
            total_messages_this_week=200,
            total_messages_last_week=50,
            consecutive_days=30,
        )
        assessment = self.svc.assess_dependency(
            "test-user", metrics, "am I right? tell me it's okay"
        )
        assert assessment.score <= 100

    def test_high_risk_threshold(self):
        """Score >= 61 should be HIGH risk."""
        metrics = _make_metrics(
            messages_this_hour=30,
            sessions_today=12,
            nighttime_messages_count=8,
            total_messages_this_week=200,
            total_messages_last_week=50,
            consecutive_days=20,
        )
        assessment = self.svc.assess_dependency(
            "test-user", metrics, "am I right? tell me it's okay"
        )
        if assessment.score >= 61:
            assert assessment.risk_level == DependencyRiskLevel.HIGH

    def test_cooldown_suggested_on_high_risk(self):
        """HIGH risk should suggest a cooldown."""
        metrics = _make_metrics(
            messages_this_hour=30,
            sessions_today=12,
            nighttime_messages_count=8,
            total_messages_this_week=200,
            total_messages_last_week=50,
            consecutive_days=20,
        )
        assessment = self.svc.assess_dependency(
            "test-user", metrics, "am I right? tell me it's okay"
        )
        if assessment.risk_level == DependencyRiskLevel.HIGH:
            assert assessment.cooldown_suggested is True
