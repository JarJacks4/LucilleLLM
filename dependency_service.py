"""
LucilleLLM - Dependency Detection & Anti-Dependency Service

Tracks interaction frequency, detects dependency patterns, and generates
compassionate boundary-setting messages. All detection uses lightweight
counters and keyword matching -- no additional LLM calls.

Firestore structure:
    interaction_metrics/{user_id}  (single document with rolling counters)

Follows the singleton pattern from other services.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Optional, List

from config import get_config
from firebase_service import get_firebase_service
from models import (
    DependencyAssessment,
    DependencyRiskLevel,
    DependencySignal,
    InteractionMetrics,
)

logger = logging.getLogger(__name__)


# ── Validation-Seeking Patterns ──────────────────────────

VALIDATION_SEEKING_PHRASES = [
    "am i right",
    "tell me i'm okay",
    "please validate",
    "do you think i'm",
    "am i being reasonable",
    "is it okay that i",
    "i need you to tell me",
    "just tell me everything is",
    "am i overreacting",
    "tell me it's going to be okay",
    "i can't handle this without you",
    "you're the only one",
    "i need to talk to you",
    "only you understand",
    "don't leave me",
    "please don't go",
    "i have no one else",
    "you're all i have",
]


# ── Minor Issue Escalation Detection ─────────────────────

MINOR_ISSUE_ESCALATION_PHRASES = [
    "this is the worst thing",
    "i can't survive this",
    "my life is over",
    "i'm falling apart",
    "everything is ruined",
    "i can't go on",
]

# Context words that combined with escalation phrases indicate minor issues
MINOR_ISSUE_CONTEXTS = [
    "traffic", "late", "coffee", "spilled", "forgot",
    "homework", "test", "exam", "parking", "wifi",
    "phone", "charger", "battery", "weather", "rain",
    "coworker", "meeting", "email", "deadline",
]


# ── Boundary-Setting Response Templates ──────────────────

GENTLE_BREAK_MESSAGES = [
    (
        "I've noticed we've been chatting quite a bit today. That's completely okay, "
        "but I want to make sure you're also taking time for yourself offline. "
        "How about a short break? Even 15 minutes of fresh air can do wonders."
    ),
    (
        "I'm always here for you, and I also want to encourage you to connect "
        "with the people in your life. Is there a friend or family member you "
        "could reach out to today?"
    ),
    (
        "You've been doing great work on yourself. Remember, real growth also "
        "happens in the quiet moments away from screens. Would you like to try "
        "a short offline self-care activity?"
    ),
]

MODERATE_BOUNDARY_MESSAGES = [
    (
        "I care about your wellbeing, and part of that means encouraging healthy "
        "boundaries -- even with me. I've noticed you've been using the app quite "
        "frequently. Consider spending some time today doing something you enjoy "
        "offline, or reaching out to a friend or counselor."
    ),
    (
        "Building resilience means gradually developing your own coping toolkit. "
        "You've already learned some great techniques here. Try practicing one of "
        "them on your own today without the app. You're more capable than you think."
    ),
]

HIGH_BOUNDARY_MESSAGES = [
    (
        "I want to be honest with you because I care. Your usage pattern suggests "
        "you might be relying on me more than is healthy. A real human connection -- "
        "a friend, family member, or professional therapist -- can offer something "
        "I cannot. I strongly encourage you to explore those options. Here are some "
        "resources to find a therapist: psychologytoday.com/find-a-therapist"
    ),
    (
        "I'm concerned about how frequently we've been talking. While I'm here to "
        "support you, I'm not a replacement for professional help or real human "
        "relationships. Please consider reaching out to a licensed therapist who "
        "can provide the ongoing support you deserve."
    ),
]

SELF_CARE_REMINDERS = [
    "Quick check-in: Have you had water recently? Taken a stretch break?",
    "Reminder: You don't need permission to take care of yourself today.",
    "Have you connected with someone in person today? Even a brief conversation counts.",
    "Remember: It's okay to put the phone down and just be present for a while.",
]

NIGHTTIME_MESSAGES = [
    (
        "I notice it's quite late. Getting good sleep is one of the most important "
        "things you can do for your mental health. Would you like to try a wind-down "
        "routine and we can pick this up tomorrow?"
    ),
    (
        "It's late and your body needs rest. Whatever is on your mind will still be "
        "here tomorrow, but you'll be better equipped to handle it after sleep. "
        "Goodnight -- take care of yourself."
    ),
]


# ── Service ──────────────────────────────────────────────

class DependencyService:
    """
    Service for interaction frequency tracking, dependency detection,
    and compassionate boundary-setting.

    Firestore structure:
        interaction_metrics/{user_id}  (single document with rolling counters)
    """

    COLLECTION = "interaction_metrics"

    def __init__(self):
        self._firebase = get_firebase_service()
        self._config = get_config()

    @property
    def db(self):
        return self._firebase.db if self._firebase else None

    # ── Metric Tracking (called on every message) ────────

    def record_interaction(
        self, user_id: str, message_text: str
    ) -> InteractionMetrics:
        """
        Record a new interaction and update rolling counters.
        Called on every /chat and /chat/stream request.
        Returns updated metrics. Fast: single Firestore read + write.
        """
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        current_hour_str = now.strftime("%Y-%m-%d-%H")
        current_hour = now.hour

        # Load existing or create fresh metrics
        metrics = self._get_or_create_metrics(user_id)

        # ── Daily reset ──
        if metrics.last_reset_date != today_str:
            # Check consecutive days: if last_reset_date was yesterday, increment
            if metrics.last_reset_date:
                try:
                    last_date = datetime.strptime(metrics.last_reset_date, "%Y-%m-%d")
                    days_gap = (now.date() - last_date.date()).days
                    if days_gap == 1:
                        metrics.consecutive_days += 1
                    elif days_gap > 1:
                        metrics.consecutive_days = 1
                except ValueError:
                    metrics.consecutive_days = 1
            else:
                metrics.consecutive_days = 1

            metrics.messages_today = 0
            metrics.sessions_today = 1  # new day = new session
            metrics.last_reset_date = today_str

        # ── Hourly reset ──
        if metrics.last_hour_reset != current_hour_str:
            metrics.messages_this_hour = 0
            metrics.last_hour_reset = current_hour_str

        # ── Weekly reset ──
        # Week starts on Monday (ISO standard)
        week_start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
        if metrics.week_start_date != week_start:
            # Roll current week to last week
            metrics.total_messages_last_week = metrics.total_messages_this_week
            metrics.total_messages_this_week = 0
            metrics.nighttime_messages_count = 0
            metrics.week_start_date = week_start

        # ── Increment counters ──
        metrics.messages_today += 1
        metrics.messages_this_hour += 1
        metrics.total_messages_this_week += 1

        # ── Nighttime tracking ──
        if self._is_nighttime(current_hour):
            metrics.nighttime_messages_count += 1

        metrics.last_message_at = now.isoformat()
        metrics.updated_at = now.isoformat()

        # ── Persist to Firestore ──
        self._save_metrics(metrics)

        return metrics

    def _get_or_create_metrics(self, user_id: str) -> InteractionMetrics:
        """Read metrics from Firestore or return fresh defaults."""
        if not self.db:
            return InteractionMetrics(user_id=user_id)

        try:
            doc_ref = self.db.collection(self.COLLECTION).document(user_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                data["user_id"] = user_id
                return InteractionMetrics(**data)
        except Exception as e:
            logger.warning(f"Failed to read interaction metrics for {user_id}: {e}")

        return InteractionMetrics(user_id=user_id)

    def _save_metrics(self, metrics: InteractionMetrics) -> bool:
        """Write metrics document to Firestore."""
        if not self.db:
            return False

        try:
            doc_ref = self.db.collection(self.COLLECTION).document(metrics.user_id)
            data = metrics.model_dump()
            data.pop("user_id", None)  # user_id is the document ID
            doc_ref.set(data, merge=True)
            return True
        except Exception as e:
            logger.warning(f"Failed to save interaction metrics for {metrics.user_id}: {e}")
            return False

    def _is_nighttime(self, hour: int) -> bool:
        """Check if the given hour falls in the nighttime window."""
        start = self._config.DEPENDENCY_NIGHTTIME_START_HOUR
        end = self._config.DEPENDENCY_NIGHTTIME_END_HOUR
        if start < end:
            return start <= hour < end
        # Handle wrap-around (e.g., 22 to 5)
        return hour >= start or hour < end

    # ── Dependency Assessment ────────────────────────────

    def assess_dependency(
        self,
        user_id: str,
        metrics: InteractionMetrics,
        message_text: str,
    ) -> DependencyAssessment:
        """
        Evaluate dependency risk from interaction metrics and message content.
        Pure computation, no API calls. Returns DependencyAssessment.

        Scoring (points, max 100):
          - messages_this_hour >= threshold:    +20
          - escalating_frequency (WoW ratio):   +20
          - sessions_today >= threshold:        +15
          - nighttime_messages >= threshold:    +15
          - consecutive_days >= threshold:      +10
          - validation_seeking keywords:        +10
          - minor_issue_escalation:             +10
        """
        score = 0
        signals: List[str] = []
        cfg = self._config
        text_lower = message_text.lower() if message_text else ""

        # Signal 1: High messages per hour (+20)
        if metrics.messages_this_hour >= cfg.DEPENDENCY_MAX_MESSAGES_PER_HOUR:
            score += 20
            signals.append(DependencySignal.HIGH_MESSAGES_PER_HOUR.value)

        # Signal 2: Escalating frequency - week-over-week (+20)
        if (
            metrics.total_messages_last_week > 0
            and metrics.total_messages_this_week > 0
        ):
            ratio = metrics.total_messages_this_week / metrics.total_messages_last_week
            if ratio >= cfg.DEPENDENCY_ESCALATION_RATIO:
                score += 20
                signals.append(DependencySignal.ESCALATING_FREQUENCY.value)

        # Signal 3: High sessions per day (+15)
        if metrics.sessions_today >= cfg.DEPENDENCY_MAX_SESSIONS_PER_DAY:
            score += 15
            signals.append(DependencySignal.HIGH_SESSIONS_PER_DAY.value)

        # Signal 4: Nighttime usage (+15)
        if metrics.nighttime_messages_count >= cfg.DEPENDENCY_NIGHTTIME_THRESHOLD:
            score += 15
            signals.append(DependencySignal.NIGHTTIME_USAGE.value)

        # Signal 5: Consecutive days of usage (+10)
        if metrics.consecutive_days >= cfg.DEPENDENCY_CONSECUTIVE_DAYS_THRESHOLD:
            score += 10
            signals.append(DependencySignal.CONSECUTIVE_DAYS_EXCESSIVE.value)

        # Signal 6: Validation-seeking keywords (+10)
        if text_lower and self._check_validation_seeking(text_lower):
            score += 10
            signals.append(DependencySignal.VALIDATION_SEEKING.value)

        # Signal 7: Minor issue escalation (+10)
        if text_lower and self._check_minor_issue_escalation(text_lower):
            score += 10
            signals.append(DependencySignal.MINOR_ISSUE_CRISIS.value)

        # Cap score at 100
        score = min(score, 100)

        # Map score to risk level
        if score >= 61:
            risk_level = DependencyRiskLevel.HIGH
        elif score >= 36:
            risk_level = DependencyRiskLevel.MODERATE
        elif score >= 16:
            risk_level = DependencyRiskLevel.LOW
        else:
            risk_level = DependencyRiskLevel.NONE

        # Select boundary message
        boundary_message = self.get_boundary_message(risk_level)

        # Add nighttime-specific message if currently nighttime and risk > NONE
        current_hour = datetime.now().hour
        if (
            self._is_nighttime(current_hour)
            and risk_level != DependencyRiskLevel.NONE
        ):
            boundary_message = self.get_nighttime_message()

        return DependencyAssessment(
            user_id=user_id,
            risk_level=risk_level,
            signals=signals,
            score=score,
            boundary_message=boundary_message,
            cooldown_suggested=(risk_level in (
                DependencyRiskLevel.MODERATE,
                DependencyRiskLevel.HIGH,
            )),
        )

    def _check_validation_seeking(self, text_lower: str) -> bool:
        """Check for emotional reliance / validation-seeking phrases."""
        for phrase in VALIDATION_SEEKING_PHRASES:
            if phrase in text_lower:
                return True
        return False

    def _check_minor_issue_escalation(self, text_lower: str) -> bool:
        """Check if user escalates minor issues to crisis-level language."""
        has_escalation = any(p in text_lower for p in MINOR_ISSUE_ESCALATION_PHRASES)
        has_minor_context = any(c in text_lower for c in MINOR_ISSUE_CONTEXTS)
        return has_escalation and has_minor_context

    # ── Boundary Message Selection ───────────────────────

    def get_boundary_message(self, risk_level: DependencyRiskLevel) -> str:
        """Select a compassionate boundary message for the given risk level."""
        if risk_level == DependencyRiskLevel.HIGH:
            return random.choice(HIGH_BOUNDARY_MESSAGES)
        elif risk_level == DependencyRiskLevel.MODERATE:
            return random.choice(MODERATE_BOUNDARY_MESSAGES)
        elif risk_level == DependencyRiskLevel.LOW:
            return random.choice(GENTLE_BREAK_MESSAGES)
        return ""

    def get_nighttime_message(self) -> str:
        """Get a nighttime-specific boundary message."""
        return random.choice(NIGHTTIME_MESSAGES)

    def get_self_care_reminder(self) -> str:
        """Get a periodic self-care reminder."""
        return random.choice(SELF_CARE_REMINDERS)

    # ── Prompt Override Text ─────────────────────────────

    def get_dependency_prompt_override(
        self, assessment: DependencyAssessment
    ) -> str:
        """
        Generate prompt text to inject anti-dependency guidance into
        the system prompt when dependency risk is elevated.
        """
        if assessment.risk_level == DependencyRiskLevel.NONE:
            return ""

        if assessment.risk_level == DependencyRiskLevel.LOW:
            return (
                "WELLBEING NOTE: The user has been chatting frequently today. "
                "Subtly encourage offline activities and real-world connections "
                "in your response. Do NOT be preachy or forceful about it."
            )
        elif assessment.risk_level == DependencyRiskLevel.MODERATE:
            return (
                "WELLBEING ALERT: The user shows signs of increasing reliance on this app. "
                "In your response, gently encourage them to:\n"
                "1. Take a break and do something offline they enjoy\n"
                "2. Reach out to a friend, family member, or therapist\n"
                "3. Practice a coping technique they've learned on their own\n"
                "Be warm and caring, not dismissive. Frame it as empowerment."
            )
        elif assessment.risk_level == DependencyRiskLevel.HIGH:
            return (
                "WELLBEING CRITICAL: The user's usage pattern indicates significant "
                "dependency on this app. You MUST:\n"
                "1. Acknowledge their feelings and validate that seeking support is good\n"
                "2. Firmly but compassionately recommend professional help (therapist, counselor)\n"
                "3. Suggest specific resources: psychologytoday.com/find-a-therapist\n"
                "4. Encourage real human connections\n"
                "5. Suggest a break from the app\n"
                "Do NOT enable continued dependency. You are a complement to, not a replacement for, real support."
            )
        return ""

    def format_dependency_for_prompt(
        self, assessment: DependencyAssessment
    ) -> str:
        """Format assessment as a block for the system prompt, including boundary message."""
        override = self.get_dependency_prompt_override(assessment)
        if not override:
            return ""

        parts = [override]
        if assessment.boundary_message:
            parts.append(
                f"Suggested boundary message to weave into your response: "
                f'"{assessment.boundary_message}"'
            )
        return "\n".join(parts)

    # ── Self-Care Reminder Logic ─────────────────────────

    def should_show_self_care_reminder(
        self, metrics: InteractionMetrics
    ) -> bool:
        """
        Decide whether to inject a periodic self-care reminder.
        Rule: every 10th message in a day.
        """
        return metrics.messages_today > 0 and metrics.messages_today % 10 == 0

    # ── Dependency Insights (for chat_agent_service tool) ─

    def get_dependency_insights(self, user_id: str) -> dict:
        """
        Return a summary of the user's interaction patterns for the LLM tool.
        Used by the 'get_wellbeing_check' tool in chat_agent_service.
        """
        metrics = self._get_or_create_metrics(user_id)
        assessment = self.assess_dependency(user_id, metrics, "")

        # Calculate week-over-week change
        if metrics.total_messages_last_week > 0:
            wow_change = (
                (metrics.total_messages_this_week / metrics.total_messages_last_week) - 1
            ) * 100
            wow_str = f"{wow_change:+.0f}%"
        else:
            wow_str = "N/A (first week)"

        return {
            "messages_today": metrics.messages_today,
            "sessions_today": metrics.sessions_today,
            "consecutive_days": metrics.consecutive_days,
            "nighttime_messages_this_week": metrics.nighttime_messages_count,
            "week_over_week_change": wow_str,
            "dependency_risk": assessment.risk_level.value,
            "dependency_score": assessment.score,
            "signals": assessment.signals,
            "cooldown_suggested": assessment.cooldown_suggested,
            "suggested_break_minutes": (
                self._config.DEPENDENCY_COOLDOWN_MINUTES
                if assessment.cooldown_suggested
                else 0
            ),
        }


# ── Singleton ────────────────────────────────────────────

_dependency_service: Optional[DependencyService] = None


def get_dependency_service() -> DependencyService:
    """Get or create DependencyService singleton."""
    global _dependency_service
    if _dependency_service is None:
        _dependency_service = DependencyService()
    return _dependency_service
