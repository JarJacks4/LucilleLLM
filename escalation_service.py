"""
LucilleLLM - Escalation & Annual Review Service (Phase 20)

Manages:
1. Human escalation queue — auto-created tickets when safety/dependency
   thresholds are crossed, with admin CRUD for triage/resolution.
2. Annual reviews — comprehensive periodic user reviews aggregating data
   from all services, with LLM-generated narrative and recommendations.

Singleton pattern consistent with other services.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from config import get_config
from models import (
    EscalationEvent,
    EscalationPriority,
    EscalationStats,
    EscalationStatus,
    EscalationTriggerType,
    AnnualReview,
    UpdateEscalationRequest,
)

logger = logging.getLogger(__name__)


class EscalationService:
    """
    Manages human escalation tickets and comprehensive annual reviews.

    Escalation auto-trigger rules:
    - CRITICAL safety event → URGENT priority
    - HIGH dependency (score >= 61) → HIGH priority
    - 3+ HIGH safety events in 7 days → HIGH priority (repeated pattern)

    Dedup: skips if user already has an active (PENDING/ACKNOWLEDGED/IN_PROGRESS) escalation.
    """

    ESCALATION_COLLECTION = "escalation_queue"
    REVIEW_COLLECTION = "annual_reviews"

    def __init__(self):
        self._config = get_config()
        self._db = None
        self._openai_client = None

        try:
            from firebase_service import get_firebase_service
            fb = get_firebase_service()
            self._db = fb.db
        except Exception as e:
            logger.warning(f"EscalationService: Firebase not available -- {e}")

        logger.info(
            f"EscalationService initialized -- "
            f"escalation_enabled={self._config.ESCALATION_ENABLED}, "
            f"review_enabled={self._config.REVIEW_ENABLED}, "
            f"db_available={self._db is not None}"
        )

    # ── OpenAI Client Injection ─────────────────────────

    def set_openai_client(self, client) -> None:
        """
        Inject the OpenAI client from main.py.
        Called after server startup when the client is initialized.
        """
        self._openai_client = client
        logger.info("EscalationService: OpenAI client injected")

    # ── Properties ──────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Check if escalation service has database access."""
        return self._db is not None

    # ══════════════════════════════════════════════════════
    #  ESCALATION QUEUE
    # ══════════════════════════════════════════════════════

    def check_and_create_escalation(
        self,
        user_id: str,
        safety_risk_level: str,
        safety_event_id: Optional[str] = None,
        dependency_risk_level: str = "NONE",
        dependency_score: int = 0,
    ) -> Optional[EscalationEvent]:
        """
        Evaluate whether an escalation ticket should be auto-created.

        Auto-trigger rules:
        1. CRITICAL safety → URGENT priority
        2. HIGH dependency (score >= 61) → HIGH priority
        3. 3+ HIGH safety events in rolling window → HIGH priority

        Returns the created EscalationEvent, or None if no escalation needed.
        """
        if not self.is_available or not self._config.ESCALATION_ENABLED:
            return None

        try:
            # Rule 1: CRITICAL safety → URGENT
            if str(safety_risk_level).upper() == "CRITICAL":
                if self._has_active_escalation(user_id):
                    logger.info(f"Escalation skipped (dedup) for user {user_id} -- CRITICAL safety")
                    return None
                return self._create_escalation(
                    user_id=user_id,
                    reason="Critical safety event detected — immediate human review required",
                    trigger_type=EscalationTriggerType.CRITICAL_SAFETY.value,
                    priority=EscalationPriority.URGENT.value,
                    safety_event_ids=[safety_event_id] if safety_event_id else [],
                    dependency_score=dependency_score,
                )

            # Rule 2: HIGH dependency (score >= 61) → HIGH
            if str(dependency_risk_level).upper() == "HIGH" and dependency_score >= 61:
                if self._has_active_escalation(user_id):
                    logger.info(f"Escalation skipped (dedup) for user {user_id} -- HIGH dependency")
                    return None
                return self._create_escalation(
                    user_id=user_id,
                    reason=f"High dependency risk detected (score: {dependency_score})",
                    trigger_type=EscalationTriggerType.HIGH_DEPENDENCY.value,
                    priority=EscalationPriority.HIGH.value,
                    safety_event_ids=[safety_event_id] if safety_event_id else [],
                    dependency_score=dependency_score,
                )

            # Rule 3: Repeated HIGH safety events in window → HIGH
            if str(safety_risk_level).upper() == "HIGH":
                if self._check_repeated_high_risk(user_id):
                    if self._has_active_escalation(user_id):
                        logger.info(f"Escalation skipped (dedup) for user {user_id} -- repeated HIGH")
                        return None
                    return self._create_escalation(
                        user_id=user_id,
                        reason=(
                            f"Repeated high-risk safety events detected "
                            f"({self._config.ESCALATION_REPEATED_HIGH_THRESHOLD}+ in "
                            f"{self._config.ESCALATION_REPEATED_HIGH_WINDOW_DAYS} days)"
                        ),
                        trigger_type=EscalationTriggerType.REPEATED_HIGH_RISK.value,
                        priority=EscalationPriority.HIGH.value,
                        safety_event_ids=[safety_event_id] if safety_event_id else [],
                        dependency_score=dependency_score,
                    )

            return None

        except Exception as e:
            logger.error(f"Escalation check failed for user {user_id}: {e}")
            return None

    def create_manual_escalation(
        self,
        user_id: str,
        reason: str,
        priority: str = EscalationPriority.NORMAL.value,
    ) -> Optional[EscalationEvent]:
        """Admin-initiated escalation (no dedup check)."""
        if not self.is_available:
            return None
        try:
            return self._create_escalation(
                user_id=user_id,
                reason=reason,
                trigger_type=EscalationTriggerType.MANUAL.value,
                priority=priority,
                safety_event_ids=[],
                dependency_score=None,
            )
        except Exception as e:
            logger.error(f"Manual escalation failed for user {user_id}: {e}")
            return None

    def _create_escalation(
        self,
        user_id: str,
        reason: str,
        trigger_type: str,
        priority: str,
        safety_event_ids: List[str],
        dependency_score: Optional[int],
    ) -> Optional[EscalationEvent]:
        """Create and persist an escalation ticket."""
        escalation_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        event = EscalationEvent(
            escalation_id=escalation_id,
            user_id=user_id,
            reason=reason,
            trigger_type=trigger_type,
            priority=priority,
            status=EscalationStatus.PENDING.value,
            safety_event_ids=safety_event_ids,
            dependency_score=dependency_score,
            notes="",
            created_at=now,
            updated_at=now,
            resolved_at=None,
            resolved_by=None,
        )

        self._db.collection(self.ESCALATION_COLLECTION).document(escalation_id).set(
            event.model_dump()
        )
        logger.warning(
            f"ESCALATION CREATED -- id={escalation_id}, user={user_id}, "
            f"type={trigger_type}, priority={priority}"
        )
        return event

    def _has_active_escalation(self, user_id: str) -> bool:
        """Check if user already has an active escalation (dedup)."""
        if not self._db:
            return False
        try:
            active_statuses = [
                EscalationStatus.PENDING.value,
                EscalationStatus.ACKNOWLEDGED.value,
                EscalationStatus.IN_PROGRESS.value,
            ]
            query = (
                self._db.collection(self.ESCALATION_COLLECTION)
                .where("user_id", "==", user_id)
                .where("status", "in", active_statuses)
                .limit(1)
            )
            docs = list(query.stream())
            return len(docs) > 0
        except Exception as e:
            logger.error(f"Active escalation check failed for user {user_id}: {e}")
            return False

    def _check_repeated_high_risk(self, user_id: str) -> bool:
        """Check if user has repeated HIGH safety events in rolling window."""
        if not self._db:
            return False
        try:
            window_start = (
                datetime.now()
                - timedelta(days=self._config.ESCALATION_REPEATED_HIGH_WINDOW_DAYS)
            ).isoformat()

            events_ref = (
                self._db.collection("safety_audit")
                .document(user_id)
                .collection("events")
                .where("risk_level", "==", "HIGH")
                .where("timestamp", ">=", window_start)
            )
            docs = list(events_ref.stream())
            count = len(docs)
            return count >= self._config.ESCALATION_REPEATED_HIGH_THRESHOLD
        except Exception as e:
            logger.error(f"Repeated high-risk check failed for user {user_id}: {e}")
            return False

    def get_escalation(self, escalation_id: str) -> Optional[dict]:
        """Get a single escalation ticket by ID."""
        if not self.is_available:
            return None
        try:
            doc = (
                self._db.collection(self.ESCALATION_COLLECTION)
                .document(escalation_id)
                .get()
            )
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            logger.error(f"Get escalation failed for {escalation_id}: {e}")
            return None

    def list_escalations(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """List escalation tickets with optional filters, newest first."""
        if not self.is_available:
            return []
        try:
            query = self._db.collection(self.ESCALATION_COLLECTION)
            if status:
                query = query.where("status", "==", status)
            if priority:
                query = query.where("priority", "==", priority)
            query = query.order_by("created_at", direction="DESCENDING").limit(limit)
            docs = list(query.stream())
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"List escalations failed: {e}")
            return []

    def update_escalation(
        self, escalation_id: str, update: UpdateEscalationRequest
    ) -> Optional[dict]:
        """Update an escalation ticket (status, notes, resolved_by)."""
        if not self.is_available:
            return None
        try:
            doc_ref = self._db.collection(self.ESCALATION_COLLECTION).document(
                escalation_id
            )
            doc = doc_ref.get()
            if not doc.exists:
                return None

            updates = {"updated_at": datetime.now().isoformat()}

            if update.status is not None:
                updates["status"] = update.status
                # Set resolved_at/resolved_by on terminal states
                if update.status in (
                    EscalationStatus.RESOLVED.value,
                    EscalationStatus.DISMISSED.value,
                ):
                    updates["resolved_at"] = datetime.now().isoformat()
                    if update.resolved_by:
                        updates["resolved_by"] = update.resolved_by

            if update.notes is not None:
                updates["notes"] = update.notes

            if update.resolved_by is not None and "resolved_by" not in updates:
                updates["resolved_by"] = update.resolved_by

            doc_ref.update(updates)

            updated_doc = doc_ref.get()
            return updated_doc.to_dict() if updated_doc.exists else None
        except Exception as e:
            logger.error(f"Update escalation failed for {escalation_id}: {e}")
            return None

    def get_escalation_stats(self) -> EscalationStats:
        """Aggregate statistics for the escalation queue."""
        if not self.is_available:
            return EscalationStats()
        try:
            docs = list(self._db.collection(self.ESCALATION_COLLECTION).stream())

            stats = {
                "total_pending": 0,
                "total_acknowledged": 0,
                "total_in_progress": 0,
                "total_resolved": 0,
                "total_dismissed": 0,
            }
            by_priority = {}
            by_trigger_type = {}
            resolution_hours = []

            for doc in docs:
                data = doc.to_dict()
                status = data.get("status", "")
                priority = data.get("priority", "")
                trigger_type = data.get("trigger_type", "")

                # Count by status
                status_key = f"total_{status}"
                if status_key in stats:
                    stats[status_key] += 1

                # Count by priority
                by_priority[priority] = by_priority.get(priority, 0) + 1

                # Count by trigger type
                by_trigger_type[trigger_type] = (
                    by_trigger_type.get(trigger_type, 0) + 1
                )

                # Avg resolution time
                if data.get("resolved_at") and data.get("created_at"):
                    try:
                        created = datetime.fromisoformat(data["created_at"])
                        resolved = datetime.fromisoformat(data["resolved_at"])
                        hours = (resolved - created).total_seconds() / 3600
                        resolution_hours.append(hours)
                    except (ValueError, TypeError):
                        pass

            avg_hours = (
                sum(resolution_hours) / len(resolution_hours)
                if resolution_hours
                else 0.0
            )

            return EscalationStats(
                total_pending=stats["total_pending"],
                total_acknowledged=stats["total_acknowledged"],
                total_in_progress=stats["total_in_progress"],
                total_resolved=stats["total_resolved"],
                total_dismissed=stats["total_dismissed"],
                by_priority=by_priority,
                by_trigger_type=by_trigger_type,
                avg_resolution_hours=round(avg_hours, 2),
            )
        except Exception as e:
            logger.error(f"Get escalation stats failed: {e}")
            return EscalationStats()

    # ══════════════════════════════════════════════════════
    #  ANNUAL REVIEWS
    # ══════════════════════════════════════════════════════

    def generate_annual_review(
        self,
        user_id: str,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Optional[AnnualReview]:
        """
        Generate a comprehensive periodic review aggregating all user data.

        Steps:
        1. Compute period defaults (last N days)
        2. Gather data from all services
        3. Generate LLM narrative + recommendations
        4. Persist and return
        """
        if not self._config.REVIEW_ENABLED:
            logger.info("Annual reviews disabled by config")
            return None

        if not self.is_available:
            return None

        if not self._openai_client:
            logger.warning("Annual review skipped — no OpenAI client available")
            return None

        try:
            # Default period
            now = datetime.now()
            if not period_end:
                period_end = now.isoformat()
            if not period_start:
                period_start = (
                    now - timedelta(days=self._config.REVIEW_DEFAULT_PERIOD_DAYS)
                ).isoformat()

            # ── Gather data from services ─────────────────

            progress_summary = self._get_progress_data(user_id)
            effectiveness_summary = self._get_effectiveness_data(user_id)
            safety_summary = self._get_safety_summary(user_id, period_start, period_end)
            health_summary = self._get_health_data(user_id)
            engagement_summary = self._get_engagement_summary(user_id)
            rl_insights = self._get_rl_insights(user_id)

            # ── Generate narrative via LLM ────────────────

            all_data = {
                "progress": progress_summary,
                "effectiveness": effectiveness_summary,
                "safety": safety_summary,
                "health": health_summary,
                "engagement": engagement_summary,
                "rl_insights": rl_insights,
            }

            narrative, recommendations = self._generate_narrative(
                all_data, period_start, period_end
            )

            # ── Build and persist review ──────────────────

            review_id = str(uuid.uuid4())
            review = AnnualReview(
                review_id=review_id,
                user_id=user_id,
                review_period_start=period_start,
                review_period_end=period_end,
                progress_summary=progress_summary,
                effectiveness_summary=effectiveness_summary,
                safety_summary=safety_summary,
                health_summary=health_summary,
                engagement_summary=engagement_summary,
                rl_insights=rl_insights,
                recommendations=recommendations,
                narrative=narrative,
                generated_at=now.isoformat(),
            )

            # Store in annual_reviews/{user_id}/reviews/{review_id}
            (
                self._db.collection(self.REVIEW_COLLECTION)
                .document(user_id)
                .collection("reviews")
                .document(review_id)
                .set(review.model_dump())
            )

            logger.info(
                f"Annual review generated -- user={user_id}, review_id={review_id}, "
                f"period={period_start[:10]}..{period_end[:10]}"
            )
            return review

        except Exception as e:
            logger.error(f"Annual review generation failed for user {user_id}: {e}")
            return None

    def _get_progress_data(self, user_id: str) -> dict:
        """Gather progress summary from progress_service."""
        try:
            from progress_service import get_progress_service
            svc = get_progress_service()
            summary = svc.get_progress_summary(user_id)
            return summary.model_dump() if summary else {}
        except Exception as e:
            logger.warning(f"Progress data unavailable for review: {e}")
            return {}

    def _get_effectiveness_data(self, user_id: str) -> dict:
        """Gather effectiveness profile from feedback_service."""
        try:
            from feedback_service import get_feedback_service
            svc = get_feedback_service()
            profile = svc.compute_effectiveness(user_id)
            return profile.model_dump() if profile else {}
        except Exception as e:
            logger.warning(f"Effectiveness data unavailable for review: {e}")
            return {}

    def _get_health_data(self, user_id: str) -> dict:
        """Gather health summary from wearable_service."""
        try:
            from wearable_service import get_wearable_service
            svc = get_wearable_service()
            summary = svc.get_health_summary(user_id)
            return summary.model_dump() if summary else {}
        except Exception as e:
            logger.warning(f"Health data unavailable for review: {e}")
            return {}

    def _get_safety_summary(
        self, user_id: str, period_start: str, period_end: str
    ) -> dict:
        """Count safety events by risk_level and event_type in period."""
        if not self._db:
            return {}
        try:
            events_ref = (
                self._db.collection("safety_audit")
                .document(user_id)
                .collection("events")
                .where("timestamp", ">=", period_start)
                .where("timestamp", "<=", period_end)
            )
            docs = list(events_ref.stream())

            by_risk_level = {}
            by_event_type = {}
            total = 0
            for doc in docs:
                data = doc.to_dict()
                total += 1
                rl = data.get("risk_level", "UNKNOWN")
                et = data.get("event_type", "unknown")
                by_risk_level[rl] = by_risk_level.get(rl, 0) + 1
                by_event_type[et] = by_event_type.get(et, 0) + 1

            return {
                "total_events": total,
                "by_risk_level": by_risk_level,
                "by_event_type": by_event_type,
            }
        except Exception as e:
            logger.warning(f"Safety summary unavailable for review: {e}")
            return {}

    def _get_engagement_summary(self, user_id: str) -> dict:
        """Read interaction_metrics for the user."""
        if not self._db:
            return {}
        try:
            doc = self._db.collection("interaction_metrics").document(user_id).get()
            return doc.to_dict() if doc.exists else {}
        except Exception as e:
            logger.warning(f"Engagement data unavailable for review: {e}")
            return {}

    def _get_rl_insights(self, user_id: str) -> dict:
        """Read bandit_state and find top arms per emotion group."""
        if not self._db:
            return {}
        try:
            doc = self._db.collection("bandit_state").document(user_id).get()
            if not doc.exists:
                return {}

            data = doc.to_dict()
            arms = data.get("arms", {})

            # Group by emotion → find top arm by success rate
            by_group = {}
            for arm_key, arm_data in arms.items():
                parts = arm_key.split("|")
                if len(parts) == 2:
                    emotion, technique = parts
                else:
                    emotion, technique = "general", arm_key

                successes = arm_data.get("successes", 0)
                failures = arm_data.get("failures", 0)
                total = successes + failures
                rate = successes / total if total > 0 else 0.0

                if emotion not in by_group or rate > by_group[emotion].get("rate", 0):
                    by_group[emotion] = {
                        "top_technique": technique,
                        "successes": successes,
                        "failures": failures,
                        "total_outcomes": total,
                        "rate": round(rate, 3),
                    }

            return {
                "total_arms": len(arms),
                "top_by_emotion": by_group,
            }
        except Exception as e:
            logger.warning(f"RL insights unavailable for review: {e}")
            return {}

    def _generate_narrative(
        self, data: dict, period_start: str, period_end: str
    ) -> Tuple[str, List[str]]:
        """
        Generate LLM narrative and recommendations from aggregated review data.

        Returns (narrative_text, list_of_recommendations).
        """
        if not self._openai_client:
            return ("Review data collected but narrative generation unavailable.", [])

        try:
            prompt = (
                "You are a mental health platform analyst generating a comprehensive "
                "periodic review for a user of an AI-assisted therapeutic companion app. "
                "Based on the following aggregated data, write:\n"
                "1. A warm, professional narrative summary (2-4 paragraphs) highlighting "
                "progress, patterns, strengths, and areas of concern.\n"
                "2. 3-5 specific, actionable recommendations.\n\n"
                f"Review period: {period_start[:10]} to {period_end[:10]}\n\n"
                f"Data:\n{json.dumps(data, indent=2, default=str)}\n\n"
                "Respond in JSON format:\n"
                '{"narrative": "...", "recommendations": ["...", "..."]}'
            )

            response = self._openai_client.chat.completions.create(
                model=self._config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500,
            )

            content = response.choices[0].message.content.strip()

            # Try to parse JSON response
            try:
                # Handle markdown code blocks
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()

                parsed = json.loads(content)
                narrative = parsed.get("narrative", content)
                recommendations = parsed.get("recommendations", [])
                return (narrative, recommendations)
            except (json.JSONDecodeError, IndexError):
                # Fallback: use raw text as narrative
                return (content, [])

        except Exception as e:
            logger.error(f"Narrative generation failed: {e}")
            return ("Review data collected but narrative generation encountered an error.", [])

    def list_reviews(self, user_id: str) -> List[dict]:
        """List all reviews for a user, newest first."""
        if not self.is_available:
            return []
        try:
            docs = (
                self._db.collection(self.REVIEW_COLLECTION)
                .document(user_id)
                .collection("reviews")
                .order_by("generated_at", direction="DESCENDING")
                .stream()
            )
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"List reviews failed for user {user_id}: {e}")
            return []

    def get_review(self, user_id: str, review_id: str) -> Optional[dict]:
        """Get a specific review by ID."""
        if not self.is_available:
            return None
        try:
            doc = (
                self._db.collection(self.REVIEW_COLLECTION)
                .document(user_id)
                .collection("reviews")
                .document(review_id)
                .get()
            )
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            logger.error(f"Get review failed for {user_id}/{review_id}: {e}")
            return None


# ── Singleton ─────────────────────────────────────────

_escalation_service: Optional[EscalationService] = None


def get_escalation_service() -> EscalationService:
    """Get or create EscalationService singleton."""
    global _escalation_service
    if _escalation_service is None:
        _escalation_service = EscalationService()
    return _escalation_service
