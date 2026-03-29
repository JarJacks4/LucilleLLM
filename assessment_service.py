"""
LucilleLLM - Assessment Service (Phase 21)

Clinically validated mental health assessments using public-domain instruments:
  - PHQ-9  (Patient Health Questionnaire-9)  — Depression screening
  - GAD-7  (Generalized Anxiety Disorder-7)  — Anxiety screening
  - WHO-5  (WHO Well-Being Index)            — Positive well-being

All scoring is deterministic arithmetic on user answers.
No AI/ML interpretation — published algorithms only.

Follows the singleton pattern from other services.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from firebase_service import get_firebase_service
from config import get_config
from models import (
    AssessmentAnswer,
    AssessmentQuestion,
    AssessmentResult,
    AssessmentSession,
    AssessmentSessionStatus,
    AssessmentSeverity,
    AssessmentType,
    ASSESSMENT_DISCLAIMER,
    RiskLevel,
    SafetyEventType,
    WellnessScore,
)

logger = logging.getLogger(__name__)


# ── Published Instrument Definitions ─────────────────────
# Question text is EXACT published wording. Do NOT modify.

PHQ9_VALUE_LABELS = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day",
}

GAD7_VALUE_LABELS = PHQ9_VALUE_LABELS  # Same scale

WHO5_VALUE_LABELS = {
    0: "At no time",
    1: "Some of the time",
    2: "Less than half of the time",
    3: "More than half of the time",
    4: "Most of the time",
    5: "All of the time",
}

PHQ9_PREAMBLE = "Over the last 2 weeks, how often have you been bothered by the following problems?"
GAD7_PREAMBLE = "Over the last 2 weeks, how often have you been bothered by the following problems?"
WHO5_PREAMBLE = "Please indicate for each of the five statements which is closest to how you have been feeling over the last two weeks."

PHQ9_QUESTIONS: List[AssessmentQuestion] = [
    AssessmentQuestion(
        index=0, text="Little interest or pleasure in doing things",
        min_value=0, max_value=3, value_labels=PHQ9_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=1, text="Feeling down, depressed, or hopeless",
        min_value=0, max_value=3, value_labels=PHQ9_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=2, text="Trouble falling or staying asleep, or sleeping too much",
        min_value=0, max_value=3, value_labels=PHQ9_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=3, text="Feeling tired or having little energy",
        min_value=0, max_value=3, value_labels=PHQ9_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=4, text="Poor appetite or overeating",
        min_value=0, max_value=3, value_labels=PHQ9_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=5,
        text="Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
        min_value=0, max_value=3, value_labels=PHQ9_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=6,
        text="Trouble concentrating on things, such as reading the newspaper or watching television",
        min_value=0, max_value=3, value_labels=PHQ9_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=7,
        text=(
            "Moving or speaking so slowly that other people could have noticed? "
            "Or the opposite — being so fidgety or restless that you have been "
            "moving around a lot more than usual"
        ),
        min_value=0, max_value=3, value_labels=PHQ9_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=8,
        text="Thoughts that you would be better off dead, or of hurting yourself",
        min_value=0, max_value=3, value_labels=PHQ9_VALUE_LABELS,
    ),
]

GAD7_QUESTIONS: List[AssessmentQuestion] = [
    AssessmentQuestion(
        index=0, text="Feeling nervous, anxious, or on edge",
        min_value=0, max_value=3, value_labels=GAD7_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=1, text="Not being able to stop or control worrying",
        min_value=0, max_value=3, value_labels=GAD7_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=2, text="Worrying too much about different things",
        min_value=0, max_value=3, value_labels=GAD7_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=3, text="Trouble relaxing",
        min_value=0, max_value=3, value_labels=GAD7_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=4, text="Being so restless that it is hard to sit still",
        min_value=0, max_value=3, value_labels=GAD7_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=5, text="Becoming easily annoyed or irritable",
        min_value=0, max_value=3, value_labels=GAD7_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=6, text="Feeling afraid, as if something awful might happen",
        min_value=0, max_value=3, value_labels=GAD7_VALUE_LABELS,
    ),
]

WHO5_QUESTIONS: List[AssessmentQuestion] = [
    AssessmentQuestion(
        index=0, text="I have felt cheerful and in good spirits",
        min_value=0, max_value=5, value_labels=WHO5_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=1, text="I have felt calm and relaxed",
        min_value=0, max_value=5, value_labels=WHO5_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=2, text="I have felt active and vigorous",
        min_value=0, max_value=5, value_labels=WHO5_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=3, text="I woke up feeling fresh and rested",
        min_value=0, max_value=5, value_labels=WHO5_VALUE_LABELS,
    ),
    AssessmentQuestion(
        index=4, text="My daily life has been filled with things that interest me",
        min_value=0, max_value=5, value_labels=WHO5_VALUE_LABELS,
    ),
]

INSTRUMENTS: Dict[AssessmentType, dict] = {
    AssessmentType.PHQ9: {
        "name": "Patient Health Questionnaire-9 (PHQ-9)",
        "description": "Screens for depression severity over the past 2 weeks.",
        "preamble": PHQ9_PREAMBLE,
        "questions": PHQ9_QUESTIONS,
        "score_range": "0-27",
        "source": "Kroenke, Spitzer & Williams, 2001. Public domain (Pfizer).",
    },
    AssessmentType.GAD7: {
        "name": "Generalized Anxiety Disorder-7 (GAD-7)",
        "description": "Screens for anxiety severity over the past 2 weeks.",
        "preamble": GAD7_PREAMBLE,
        "questions": GAD7_QUESTIONS,
        "score_range": "0-21",
        "source": "Spitzer, Kroenke, Williams & Lowe, 2006. Public domain.",
    },
    AssessmentType.WHO5: {
        "name": "WHO-5 Well-Being Index",
        "description": "Measures subjective psychological well-being over the past 2 weeks.",
        "preamble": WHO5_PREAMBLE,
        "questions": WHO5_QUESTIONS,
        "score_range": "0-100 (raw 0-25 × 4)",
        "source": "World Health Organization, 1998. Public domain.",
    },
}


# ── Scoring Functions (Pure, Deterministic) ──────────────
# These implement the EXACT published algorithms.
# No custom weights, no AI, no ML. Just arithmetic.


def _classify_phq9(score: int) -> Tuple[AssessmentSeverity, str]:
    """PHQ-9 severity classification per Kroenke et al. 2001."""
    if score <= 4:
        return (AssessmentSeverity.MINIMAL, "Minimal Depression")
    if score <= 9:
        return (AssessmentSeverity.MILD, "Mild Depression")
    if score <= 14:
        return (AssessmentSeverity.MODERATE, "Moderate Depression")
    if score <= 19:
        return (AssessmentSeverity.MODERATELY_SEVERE, "Moderately Severe Depression")
    return (AssessmentSeverity.SEVERE, "Severe Depression")


def _classify_gad7(score: int) -> Tuple[AssessmentSeverity, str]:
    """GAD-7 severity classification per Spitzer et al. 2006."""
    if score <= 4:
        return (AssessmentSeverity.MINIMAL, "Minimal Anxiety")
    if score <= 9:
        return (AssessmentSeverity.MILD, "Mild Anxiety")
    if score <= 14:
        return (AssessmentSeverity.MODERATE, "Moderate Anxiety")
    return (AssessmentSeverity.SEVERE, "Severe Anxiety")


def _classify_who5(scaled_score: int) -> Tuple[AssessmentSeverity, str]:
    """WHO-5 well-being classification. Higher = better (inverted vs PHQ-9/GAD-7)."""
    if scaled_score >= 75:
        return (AssessmentSeverity.MINIMAL, "Good Well-being")
    if scaled_score >= 50:
        return (AssessmentSeverity.MILD, "Moderate Well-being")
    if scaled_score >= 25:
        return (AssessmentSeverity.MODERATE, "Low Well-being")
    return (AssessmentSeverity.SEVERE, "Poor Well-being")


# ── Assessment Service ───────────────────────────────────


class AssessmentService:
    """
    Service for clinically validated mental health assessments.

    Uses PHQ-9, GAD-7, and WHO-5 — all public domain instruments
    with published scoring algorithms. No AI/ML interpretation.
    """

    COLLECTION = "assessments"

    def __init__(self):
        self._firebase = get_firebase_service()
        self._config = get_config()

    @property
    def db(self):
        return self._firebase.db

    # ── Instrument Metadata ──────────────────────────────

    def get_instruments(self) -> List[dict]:
        """List available assessment instruments with metadata."""
        result = []
        for atype, info in INSTRUMENTS.items():
            result.append({
                "assessment_type": atype.value,
                "name": info["name"],
                "description": info["description"],
                "preamble": info["preamble"],
                "question_count": len(info["questions"]),
                "score_range": info["score_range"],
                "source": info["source"],
            })
        return result

    def get_instrument_questions(
        self, assessment_type: AssessmentType
    ) -> List[AssessmentQuestion]:
        """Return the questions for a specific instrument."""
        return INSTRUMENTS[assessment_type]["questions"]

    # ── Session Management ───────────────────────────────

    def start_assessment(
        self,
        user_id: str,
        assessment_type: AssessmentType,
    ) -> Optional[AssessmentSession]:
        """Start a new assessment session. Returns None if feature disabled."""
        if not self._config.ASSESSMENT_ENABLED:
            return None

        questions = INSTRUMENTS[assessment_type]["questions"]

        session = AssessmentSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            assessment_type=assessment_type,
            total_questions=len(questions),
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
                    f"Started {assessment_type.value} assessment for user {user_id}, "
                    f"session {session.session_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to store assessment session: {e}")

        return session

    def get_in_progress_session(
        self,
        user_id: str,
        assessment_type: AssessmentType,
    ) -> Optional[AssessmentSession]:
        """Check if user has an in-progress session for this instrument."""
        if self.db is None:
            return None

        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("sessions")
            )
            query = (
                col_ref.where("assessment_type", "==", assessment_type.value)
                .where("status", "==", AssessmentSessionStatus.IN_PROGRESS.value)
                .limit(1)
            )
            for doc in query.stream():
                data = doc.to_dict()
                return AssessmentSession(**data)
            return None
        except Exception as e:
            logger.warning(f"Failed to check in-progress session: {e}")
            return None

    # ── Answer Submission ────────────────────────────────

    def submit_answer(
        self,
        user_id: str,
        session_id: str,
        value: int,
    ) -> Tuple[Optional[AssessmentSession], Optional[AssessmentQuestion], Optional[str]]:
        """
        Submit an answer to the current question.

        Returns (updated_session, next_question_or_None, safety_notice_or_None).
        Returns (None, None, None) on error.
        """
        if self.db is None:
            return (None, None, None)

        try:
            doc_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("sessions")
                .document(session_id)
            )
            doc = doc_ref.get()
            if not doc.exists:
                return (None, None, None)

            data = doc.to_dict()
            session = AssessmentSession(**data)

            if session.status != AssessmentSessionStatus.IN_PROGRESS:
                return (session, None, None)

            # Get instrument questions
            questions = INSTRUMENTS[session.assessment_type]["questions"]
            current_q = questions[session.current_question_index]

            # Validate value range for this specific instrument
            if value < current_q.min_value or value > current_q.max_value:
                logger.warning(
                    f"Invalid answer value {value} for {session.assessment_type.value} "
                    f"Q{session.current_question_index} (range {current_q.min_value}-{current_q.max_value})"
                )
                return (session, None, None)

            # Record answer
            answer = AssessmentAnswer(
                question_index=session.current_question_index,
                value=value,
            )
            session.answers.append(answer)

            # ── SAFETY CHECK: PHQ-9 Question 9 (self-harm ideation) ──
            safety_notice = None
            if (
                session.assessment_type == AssessmentType.PHQ9
                and session.current_question_index == 8  # Q9, 0-indexed
                and value > 0
            ):
                session.safety_flagged = True
                safety_notice = self._handle_phq9_q9_safety(
                    user_id, session_id, value
                )

            # Advance to next question
            session.current_question_index += 1

            # Determine next question
            next_question = None
            if session.current_question_index < session.total_questions:
                next_question = questions[session.current_question_index]

            # Update Firestore
            doc_ref.update({
                "answers": [a.model_dump() for a in session.answers],
                "current_question_index": session.current_question_index,
                "safety_flagged": session.safety_flagged,
            })

            return (session, next_question, safety_notice)

        except Exception as e:
            logger.warning(f"Failed to submit assessment answer: {e}")
            return (None, None, None)

    def _handle_phq9_q9_safety(
        self, user_id: str, session_id: str, value: int
    ) -> str:
        """
        Handle PHQ-9 Question 9 self-harm ideation response.
        Logs safety event + creates escalation. Returns crisis notice text.
        """
        safety_notice = ""

        try:
            from safety_service import get_safety_service
            safety_svc = get_safety_service()

            # Log safety event
            event_id = safety_svc.log_safety_event(
                user_id=user_id,
                session_id=session_id,
                event_type=SafetyEventType.HIGH_RISK_INPUT,
                risk_level=RiskLevel.HIGH,
                message_snippet=f"PHQ-9 Q9 self-harm ideation score: {value}/3",
                flags=["phq9_q9_self_harm_ideation", f"q9_score:{value}"],
                action_taken="assessment_safety_flag",
            )

            # Get crisis intercept text
            safety_notice = safety_svc.get_crisis_intercept_response()

            # Trigger escalation
            try:
                from escalation_service import get_escalation_service
                esc_svc = get_escalation_service()
                esc_svc.check_and_create_escalation(
                    user_id=user_id,
                    safety_risk_level=RiskLevel.HIGH.value,
                    safety_event_id=event_id,
                    dependency_risk_level="NONE",
                    dependency_score=0,
                )
            except Exception as e:
                logger.warning(f"Failed to create escalation for PHQ-9 Q9: {e}")

            logger.warning(
                f"PHQ-9 Q9 SAFETY FLAG: user {user_id}, value={value}/3, "
                f"event_id={event_id}"
            )

        except Exception as e:
            logger.error(f"Failed to handle PHQ-9 Q9 safety: {e}")
            # Fallback safety notice
            safety_notice = (
                "If you are having thoughts of harming yourself, please reach out "
                "for help immediately. Call or text 988 (Suicide & Crisis Lifeline) "
                "or text HOME to 741741 (Crisis Text Line)."
            )

        return safety_notice

    # ── Scoring & Completion ─────────────────────────────

    def complete_assessment(
        self,
        user_id: str,
        session_id: str,
    ) -> Optional[AssessmentSession]:
        """
        Complete an assessment and compute scores.

        Scoring uses EXACT published algorithms:
          PHQ-9: sum of 9 items (0-27)
          GAD-7: sum of 7 items (0-21)
          WHO-5: sum of 5 items (0-25) × 4 = 0-100

        Returns None on error.
        """
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
            session = AssessmentSession(**data)

            if session.status != AssessmentSessionStatus.IN_PROGRESS:
                return session

            # Validate all questions answered
            if len(session.answers) < session.total_questions:
                logger.warning(
                    f"Cannot complete: {len(session.answers)}/{session.total_questions} "
                    f"questions answered"
                )
                return session

            # Compute raw score: sum of all answer values
            raw_score = sum(a.value for a in session.answers)

            # Classify based on instrument type
            scaled_score = None
            if session.assessment_type == AssessmentType.PHQ9:
                severity, severity_label = _classify_phq9(raw_score)
            elif session.assessment_type == AssessmentType.GAD7:
                severity, severity_label = _classify_gad7(raw_score)
            else:  # WHO5
                scaled_score = raw_score * 4
                severity, severity_label = _classify_who5(scaled_score)

            # Build concern flags
            concern_flags = self._compute_concern_flags(
                session.assessment_type, raw_score, scaled_score, session.safety_flagged
            )

            # Create result
            result = AssessmentResult(
                assessment_type=session.assessment_type,
                raw_score=raw_score,
                scaled_score=scaled_score,
                severity=severity,
                severity_label=severity_label,
                concern_flags=concern_flags,
            )

            # Update session
            session.status = AssessmentSessionStatus.COMPLETED
            session.completed_at = datetime.now().isoformat()
            session.result = result

            doc_ref.update({
                "status": session.status.value,
                "completed_at": session.completed_at,
                "result": result.model_dump(),
            })

            logger.info(
                f"Completed {session.assessment_type.value} for user {user_id}: "
                f"raw={raw_score}, severity={severity.value}, flags={concern_flags}"
            )

            return session

        except Exception as e:
            logger.warning(f"Failed to complete assessment: {e}")
            return None

    def _compute_concern_flags(
        self,
        assessment_type: AssessmentType,
        raw_score: int,
        scaled_score: Optional[int],
        safety_flagged: bool,
    ) -> List[str]:
        """Compute concern flags based on published thresholds."""
        flags = []
        config = self._config

        if assessment_type == AssessmentType.PHQ9:
            if raw_score >= config.ASSESSMENT_PHQ9_CONCERN_THRESHOLD:
                flags.append("elevated_depression")
            if raw_score >= 15:
                flags.append("high_severity_concern")

        elif assessment_type == AssessmentType.GAD7:
            if raw_score >= config.ASSESSMENT_GAD7_CONCERN_THRESHOLD:
                flags.append("elevated_anxiety")
            if raw_score >= 15:
                flags.append("high_severity_concern")

        elif assessment_type == AssessmentType.WHO5 and scaled_score is not None:
            if scaled_score < config.ASSESSMENT_WHO5_CONCERN_THRESHOLD:
                flags.append("low_wellbeing")

        if safety_flagged:
            flags.append("self_harm_ideation")

        return flags

    # ── History & Scores ─────────────────────────────────

    def get_assessment_history(
        self,
        user_id: str,
        assessment_type: Optional[AssessmentType] = None,
        limit: int = 20,
    ) -> List[dict]:
        """Get past completed assessments, optionally filtered by type."""
        if self.db is None:
            return []

        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("sessions")
            )

            query = col_ref.where(
                "status", "==", AssessmentSessionStatus.COMPLETED.value
            )
            if assessment_type is not None:
                query = query.where(
                    "assessment_type", "==", assessment_type.value
                )

            query = query.order_by(
                "completed_at", direction="DESCENDING"
            ).limit(limit)

            results = []
            for doc in query.stream():
                data = doc.to_dict()
                results.append(data)

            return results

        except Exception as e:
            logger.warning(f"Failed to get assessment history: {e}")
            return []

    def get_latest_scores(self, user_id: str) -> Dict[str, Optional[dict]]:
        """Get the most recent completed result for each instrument type."""
        scores = {}
        for atype in AssessmentType:
            history = self.get_assessment_history(
                user_id, assessment_type=atype, limit=1
            )
            if history:
                scores[atype.value] = history[0].get("result")
            else:
                scores[atype.value] = None
        return scores

    def get_wellness_score(self, user_id: str) -> WellnessScore:
        """
        Build composite wellness view.

        Primary score = WHO-5 scaled (0-100, higher = better).
        Breakdowns = PHQ-9 (depression) + GAD-7 (anxiety).
        """
        latest = self.get_latest_scores(user_id)
        concern_flags = []
        last_dates = {}

        # WHO-5 (primary)
        overall_score = None
        overall_label = "No assessment yet"
        who5_raw = None
        who5_scaled = None
        who5_result = latest.get("who5")
        if who5_result:
            who5_raw = who5_result.get("raw_score")
            who5_scaled = who5_result.get("scaled_score")
            overall_score = who5_scaled
            overall_label = who5_result.get("severity_label", "")
            concern_flags.extend(who5_result.get("concern_flags", []))

        # PHQ-9 (breakdown)
        phq9_score = None
        phq9_severity = None
        phq9_result = latest.get("phq9")
        if phq9_result:
            phq9_score = phq9_result.get("raw_score")
            phq9_severity = phq9_result.get("severity_label")
            concern_flags.extend(phq9_result.get("concern_flags", []))

        # GAD-7 (breakdown)
        gad7_score = None
        gad7_severity = None
        gad7_result = latest.get("gad7")
        if gad7_result:
            gad7_score = gad7_result.get("raw_score")
            gad7_severity = gad7_result.get("severity_label")
            concern_flags.extend(gad7_result.get("concern_flags", []))

        # Last assessment dates
        for atype in AssessmentType:
            history = self.get_assessment_history(
                user_id, assessment_type=atype, limit=1
            )
            if history:
                last_dates[atype.value] = history[0].get("completed_at", "")

        # Deduplicate flags
        concern_flags = list(dict.fromkeys(concern_flags))

        return WellnessScore(
            user_id=user_id,
            overall_score=overall_score,
            overall_label=overall_label,
            phq9_score=phq9_score,
            phq9_severity=phq9_severity,
            gad7_score=gad7_score,
            gad7_severity=gad7_severity,
            who5_raw=who5_raw,
            who5_scaled=who5_scaled,
            concern_flags=concern_flags,
            last_assessment_dates=last_dates,
        )


# ── Singleton ────────────────────────────────────────────

_assessment_service: Optional[AssessmentService] = None


def get_assessment_service() -> AssessmentService:
    """Get or create AssessmentService singleton."""
    global _assessment_service
    if _assessment_service is None:
        _assessment_service = AssessmentService()
    return _assessment_service
