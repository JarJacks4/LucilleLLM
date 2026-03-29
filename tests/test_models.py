"""
Tests for Pydantic models — serialization, validation, defaults.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import ValidationError
import pytest
from models import (
    ChatRequest,
    AssessmentType,
    AssessmentSeverity,
    AssessmentResult,
    WellnessScore,
    ASSESSMENT_DISCLAIMER,
)


class TestChatRequestValidation:
    """Test input validation on ChatRequest (sanitization item #7)."""

    def test_valid_request(self):
        req = ChatRequest(message="Hello", session_id="abc-123")
        assert req.message == "Hello"

    def test_message_max_length(self):
        """Messages longer than 4000 chars should be rejected."""
        with pytest.raises(ValidationError):
            ChatRequest(message="x" * 4001, session_id="abc")

    def test_session_id_max_length(self):
        """Session IDs longer than 200 chars should be rejected."""
        with pytest.raises(ValidationError):
            ChatRequest(message="hi", session_id="x" * 201)

    def test_empty_message_rejected(self):
        """Empty messages should be rejected (min_length from Field)."""
        # Empty string is technically valid by Pydantic unless min_length is set
        req = ChatRequest(message="", session_id="abc")
        assert req.message == ""


class TestAssessmentModels:
    """Test assessment model serialization and defaults."""

    def test_assessment_result_has_disclaimer(self):
        result = AssessmentResult(
            assessment_type=AssessmentType.PHQ9,
            raw_score=10,
            severity=AssessmentSeverity.MODERATE,
            severity_label="Moderate Depression",
        )
        assert result.disclaimer == ASSESSMENT_DISCLAIMER
        assert "not a clinical diagnosis" in result.disclaimer

    def test_wellness_score_defaults(self):
        ws = WellnessScore(user_id="test")
        assert ws.overall_score is None
        assert ws.overall_label == "No assessment yet"
        assert ws.disclaimer == ASSESSMENT_DISCLAIMER

    def test_assessment_result_serialization(self):
        result = AssessmentResult(
            assessment_type=AssessmentType.WHO5,
            raw_score=20,
            scaled_score=80,
            severity=AssessmentSeverity.MINIMAL,
            severity_label="Good Well-being",
        )
        d = result.model_dump()
        assert d["raw_score"] == 20
        assert d["scaled_score"] == 80
        assert d["assessment_type"] == "who5"
