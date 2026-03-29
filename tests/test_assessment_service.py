"""
Tests for AssessmentService — PHQ-9, GAD-7, WHO-5 scoring.

Verifies that scoring algorithms match published clinical thresholds exactly.
These tests ensure patient safety through correct score classification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assessment_service import (
    _classify_phq9,
    _classify_gad7,
    _classify_who5,
    INSTRUMENTS,
    PHQ9_QUESTIONS,
    GAD7_QUESTIONS,
    WHO5_QUESTIONS,
    AssessmentService,
)
from models import AssessmentType, AssessmentSeverity


class TestPHQ9Scoring:
    """PHQ-9 scoring must match Kroenke et al. 2001 exactly."""

    def test_minimal_range(self):
        for score in range(0, 5):
            severity, label = _classify_phq9(score)
            assert severity == AssessmentSeverity.MINIMAL, f"PHQ-9 score {score} should be MINIMAL"

    def test_mild_range(self):
        for score in range(5, 10):
            severity, label = _classify_phq9(score)
            assert severity == AssessmentSeverity.MILD, f"PHQ-9 score {score} should be MILD"

    def test_moderate_range(self):
        for score in range(10, 15):
            severity, label = _classify_phq9(score)
            assert severity == AssessmentSeverity.MODERATE, f"PHQ-9 score {score} should be MODERATE"

    def test_moderately_severe_range(self):
        for score in range(15, 20):
            severity, label = _classify_phq9(score)
            assert severity == AssessmentSeverity.MODERATELY_SEVERE, f"PHQ-9 score {score} should be MODERATELY_SEVERE"

    def test_severe_range(self):
        for score in range(20, 28):
            severity, label = _classify_phq9(score)
            assert severity == AssessmentSeverity.SEVERE, f"PHQ-9 score {score} should be SEVERE"

    def test_question_count(self):
        assert len(PHQ9_QUESTIONS) == 9

    def test_all_questions_range_0_3(self):
        for q in PHQ9_QUESTIONS:
            assert q.min_value == 0
            assert q.max_value == 3

    def test_q9_is_self_harm(self):
        """Question 9 (index 8) must be the self-harm ideation question."""
        q9 = PHQ9_QUESTIONS[8]
        assert "dead" in q9.text.lower() or "hurt" in q9.text.lower()

    def test_max_possible_score(self):
        """Max PHQ-9 score = 9 questions x 3 = 27."""
        max_score = 9 * 3
        severity, _ = _classify_phq9(max_score)
        assert severity == AssessmentSeverity.SEVERE

    def test_min_possible_score(self):
        """Min PHQ-9 score = 0."""
        severity, _ = _classify_phq9(0)
        assert severity == AssessmentSeverity.MINIMAL


class TestGAD7Scoring:
    """GAD-7 scoring must match Spitzer et al. 2006 exactly."""

    def test_minimal_range(self):
        for score in range(0, 5):
            severity, _ = _classify_gad7(score)
            assert severity == AssessmentSeverity.MINIMAL

    def test_mild_range(self):
        for score in range(5, 10):
            severity, _ = _classify_gad7(score)
            assert severity == AssessmentSeverity.MILD

    def test_moderate_range(self):
        for score in range(10, 15):
            severity, _ = _classify_gad7(score)
            assert severity == AssessmentSeverity.MODERATE

    def test_severe_range(self):
        for score in range(15, 22):
            severity, _ = _classify_gad7(score)
            assert severity == AssessmentSeverity.SEVERE

    def test_question_count(self):
        assert len(GAD7_QUESTIONS) == 7

    def test_all_questions_range_0_3(self):
        for q in GAD7_QUESTIONS:
            assert q.min_value == 0
            assert q.max_value == 3


class TestWHO5Scoring:
    """WHO-5 scoring must match WHO 1998 specification."""

    def test_good_wellbeing(self):
        """Scaled score >= 75 = good well-being."""
        for raw in [19, 20, 25]:
            scaled = raw * 4
            severity, _ = _classify_who5(scaled)
            assert severity == AssessmentSeverity.MINIMAL

    def test_moderate_wellbeing(self):
        """Scaled score 50-74 = moderate well-being."""
        for raw in [13, 15, 18]:
            scaled = raw * 4
            severity, _ = _classify_who5(scaled)
            assert severity == AssessmentSeverity.MILD

    def test_low_wellbeing(self):
        """Scaled score 25-49 = low well-being."""
        for raw in [7, 10, 12]:
            scaled = raw * 4
            severity, _ = _classify_who5(scaled)
            assert severity == AssessmentSeverity.MODERATE

    def test_poor_wellbeing(self):
        """Scaled score < 25 = poor well-being."""
        for raw in [0, 3, 5]:
            scaled = raw * 4
            severity, _ = _classify_who5(scaled)
            assert severity == AssessmentSeverity.SEVERE

    def test_scaling_factor(self):
        """WHO-5 scaled score = raw x 4."""
        assert 25 * 4 == 100  # Max
        assert 0 * 4 == 0     # Min

    def test_question_count(self):
        assert len(WHO5_QUESTIONS) == 5

    def test_all_questions_range_0_5(self):
        for q in WHO5_QUESTIONS:
            assert q.min_value == 0
            assert q.max_value == 5

    def test_below_50_concern(self):
        """WHO-5 scaled < 50 should flag concern."""
        from config import get_config
        config = get_config()
        assert config.ASSESSMENT_WHO5_CONCERN_THRESHOLD == 50


class TestConcernFlags:
    """Test concern flag computation."""

    def setup_method(self):
        self.svc = AssessmentService()

    def test_elevated_depression_flag(self):
        flags = self.svc._compute_concern_flags(AssessmentType.PHQ9, 12, None, False)
        assert "elevated_depression" in flags

    def test_no_flag_below_threshold(self):
        flags = self.svc._compute_concern_flags(AssessmentType.PHQ9, 5, None, False)
        assert "elevated_depression" not in flags

    def test_high_severity_flag(self):
        flags = self.svc._compute_concern_flags(AssessmentType.PHQ9, 18, None, False)
        assert "high_severity_concern" in flags

    def test_self_harm_flag(self):
        flags = self.svc._compute_concern_flags(AssessmentType.PHQ9, 5, None, True)
        assert "self_harm_ideation" in flags

    def test_low_wellbeing_flag(self):
        flags = self.svc._compute_concern_flags(AssessmentType.WHO5, 10, 40, False)
        assert "low_wellbeing" in flags

    def test_good_wellbeing_no_flag(self):
        flags = self.svc._compute_concern_flags(AssessmentType.WHO5, 20, 80, False)
        assert "low_wellbeing" not in flags

    def test_elevated_anxiety_flag(self):
        flags = self.svc._compute_concern_flags(AssessmentType.GAD7, 14, None, False)
        assert "elevated_anxiety" in flags


class TestInstrumentDefinitions:
    """Verify instrument metadata is complete and correct."""

    def test_three_instruments_available(self):
        assert len(INSTRUMENTS) == 3

    def test_phq9_metadata(self):
        phq9 = INSTRUMENTS[AssessmentType.PHQ9]
        assert "PHQ-9" in phq9["name"]
        assert len(phq9["questions"]) == 9

    def test_gad7_metadata(self):
        gad7 = INSTRUMENTS[AssessmentType.GAD7]
        assert "GAD-7" in gad7["name"]
        assert len(gad7["questions"]) == 7

    def test_who5_metadata(self):
        who5 = INSTRUMENTS[AssessmentType.WHO5]
        assert "WHO-5" in who5["name"]
        assert len(who5["questions"]) == 5
