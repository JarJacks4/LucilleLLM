"""
LucilleLLM - Compliance Service (GDPR/HIPAA)

Provides data portability (export), right to erasure (cascade deletion),
consent management, and data retention policy enforcement.

Firestore structure for consent:
    consent_records/{user_id}  (single document per user)

Follows the singleton pattern from other services.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List

from config import get_config
from firebase_service import get_firebase_service
from audit_service import get_audit_service
from cache import get_cache, user_profile_key, effectiveness_key
from models import (
    AuditAction,
    ConsentRecord,
    DataExportResponse,
    DeletionReceipt,
)

logger = logging.getLogger(__name__)


class ComplianceService:
    """
    Service for GDPR/HIPAA compliance operations.

    - Data export (GDPR Art. 20 — Right to Portability)
    - Cascade deletion (GDPR Art. 17 — Right to Erasure)
    - Consent management
    - Data retention policy enforcement
    """

    CONSENT_COLLECTION = "consent_records"

    def __init__(self):
        self._firebase = get_firebase_service()
        self._config = get_config()

    @property
    def db(self):
        return self._firebase.db if self._firebase else None

    # ── Data Export (GDPR Art. 20) ────────────────────────

    def export_user_data(
        self, user_id: str, ip_address: str = ""
    ) -> DataExportResponse:
        """
        Export ALL user data as a structured response.
        Reads from all 10 collections + consent.
        Strips embedding vectors from memories (too large for export).
        Logs the export to audit trail.
        """
        if not self.db:
            return DataExportResponse(user_id=user_id, status="no_database")

        # 1. User profile
        profile = None
        try:
            doc = self.db.collection("user_profiles").document(user_id).get()
            if doc.exists:
                profile = doc.to_dict()
        except Exception as e:
            logger.warning(f"Export: failed to read user_profiles: {e}")

        # 2. Memories (strip embeddings)
        memories = self._read_subcollection(
            f"user_memories/{user_id}", "memories"
        )
        for mem in memories:
            mem.pop("embedding", None)

        # 3. Chat sessions (query by user_id field)
        chat_sessions = []
        try:
            docs = (
                self.db.collection("chat_sessions")
                .where("user_id", "==", user_id)
                .stream()
            )
            for doc in docs:
                data = doc.to_dict()
                data["session_id"] = doc.id
                chat_sessions.append(data)
        except Exception as e:
            logger.warning(f"Export: failed to read chat_sessions: {e}")

        # 4. Response feedback
        response_feedback = self._read_subcollection(
            f"feedback/{user_id}", "response_feedback"
        )

        # 5. Exercise outcomes
        exercise_outcomes = self._read_subcollection(
            f"feedback/{user_id}", "exercise_outcomes"
        )

        # 6. Exercise sessions
        exercise_sessions = self._read_subcollection(
            f"exercise_sessions/{user_id}", "sessions"
        )

        # 7. Practice tasks
        practice_tasks = self._read_subcollection(
            f"exercise_sessions/{user_id}", "tasks"
        )

        # 8. Soundscape sessions
        soundscape_sessions = self._read_subcollection(
            f"soundscape_sessions/{user_id}", "sessions"
        )

        # 9. Safety events
        safety_events = self._read_subcollection(
            f"safety_audit/{user_id}", "events"
        )

        # 10. Health metrics (wearable data)
        health_metrics = self._read_subcollection(
            f"health_metrics/{user_id}", "daily"
        )

        # 12. Bandit state (RL / Thompson Sampling)
        bandit_state = None
        try:
            doc = self.db.collection("bandit_state").document(user_id).get()
            if doc.exists:
                bandit_state = doc.to_dict()
        except Exception as e:
            logger.warning(f"Export: failed to read bandit_state: {e}")

        # 13. Training examples (Phase 18: Fine-tuning)
        training_examples = []
        try:
            docs = (
                self.db.collection("training_examples")
                .where("user_id", "==", user_id)
                .stream()
            )
            for doc in docs:
                data = doc.to_dict()
                data["_doc_id"] = doc.id
                training_examples.append(data)
        except Exception as e:
            logger.warning(f"Export: failed to read training_examples: {e}")

        # 13b. Model performance records (Phase 18)
        model_performance = []
        try:
            docs = (
                self.db.collection("model_performance")
                .where("user_id", "==", user_id)
                .stream()
            )
            for doc in docs:
                data = doc.to_dict()
                data["_doc_id"] = doc.id
                model_performance.append(data)
        except Exception as e:
            logger.warning(f"Export: failed to read model_performance: {e}")

        # 14. Interaction metrics
        interaction_metrics = None
        try:
            doc = self.db.collection("interaction_metrics").document(user_id).get()
            if doc.exists:
                interaction_metrics = doc.to_dict()
        except Exception as e:
            logger.warning(f"Export: failed to read interaction_metrics: {e}")

        # 15. Consent
        consent = None
        try:
            doc = self.db.collection(self.CONSENT_COLLECTION).document(user_id).get()
            if doc.exists:
                consent = doc.to_dict()
        except Exception as e:
            logger.warning(f"Export: failed to read consent_records: {e}")

        # 16. Escalation events (Phase 20)
        escalation_events = []
        try:
            docs = (
                self.db.collection("escalation_queue")
                .where("user_id", "==", user_id)
                .stream()
            )
            for doc in docs:
                data = doc.to_dict()
                data["_doc_id"] = doc.id
                escalation_events.append(data)
        except Exception as e:
            logger.warning(f"Export: failed to read escalation_queue: {e}")

        # 17. Annual reviews (Phase 20)
        annual_reviews = self._read_subcollection(
            f"annual_reviews/{user_id}", "reviews"
        )

        # Log the export to audit trail
        audit_svc = get_audit_service()
        audit_svc.log(
            user_id=user_id,
            action=AuditAction.EXPORT,
            resource_type="all_data",
            details="GDPR data portability export",
            actor_id=user_id,
            ip_address=ip_address,
        )

        return DataExportResponse(
            user_id=user_id,
            profile=profile,
            memories=memories,
            chat_sessions=chat_sessions,
            response_feedback=response_feedback,
            exercise_outcomes=exercise_outcomes,
            exercise_sessions=exercise_sessions,
            practice_tasks=practice_tasks,
            soundscape_sessions=soundscape_sessions,
            safety_events=safety_events,
            health_metrics=health_metrics,
            bandit_state=bandit_state,
            training_examples=training_examples,
            model_performance=model_performance,
            interaction_metrics=interaction_metrics,
            consent=consent,
            escalation_events=escalation_events,
            annual_reviews=annual_reviews,
        )

    # ── Cascade Deletion (GDPR Art. 17) ──────────────────

    def delete_all_user_data(
        self, user_id: str, ip_address: str = ""
    ) -> DeletionReceipt:
        """
        Cascade delete ALL user data across all collections.
        Satisfies GDPR Article 17 (Right to Erasure).

        IMPORTANT: audit_log entries are NOT deleted (legal requirement).
        """
        if not self.db:
            return DeletionReceipt(user_id=user_id, status="no_database")

        # Log deletion intent BEFORE deleting anything
        audit_svc = get_audit_service()
        audit_svc.log(
            user_id=user_id,
            action=AuditAction.DELETE,
            resource_type="all_data",
            details="GDPR right to erasure: cascade deletion initiated",
            actor_id=user_id,
            ip_address=ip_address,
        )

        counts = {}

        # 1. Delete subcollection: user_memories/{user_id}/memories/*
        counts["memories"] = self._delete_subcollection(
            f"user_memories/{user_id}", "memories"
        )
        self._delete_parent_doc("user_memories", user_id)

        # 2. Delete subcollection: feedback/{user_id}/response_feedback/*
        counts["response_feedback"] = self._delete_subcollection(
            f"feedback/{user_id}", "response_feedback"
        )

        # 3. Delete subcollection: feedback/{user_id}/exercise_outcomes/*
        counts["exercise_outcomes"] = self._delete_subcollection(
            f"feedback/{user_id}", "exercise_outcomes"
        )
        self._delete_parent_doc("feedback", user_id)

        # 4. Delete subcollection: exercise_sessions/{user_id}/sessions/*
        counts["exercise_sessions"] = self._delete_subcollection(
            f"exercise_sessions/{user_id}", "sessions"
        )

        # 5. Delete subcollection: exercise_sessions/{user_id}/tasks/*
        counts["practice_tasks"] = self._delete_subcollection(
            f"exercise_sessions/{user_id}", "tasks"
        )
        self._delete_parent_doc("exercise_sessions", user_id)

        # 6. Delete subcollection: soundscape_sessions/{user_id}/sessions/*
        counts["soundscape_sessions"] = self._delete_subcollection(
            f"soundscape_sessions/{user_id}", "sessions"
        )
        self._delete_parent_doc("soundscape_sessions", user_id)

        # 7. Delete subcollection: safety_audit/{user_id}/events/*
        counts["safety_events"] = self._delete_subcollection(
            f"safety_audit/{user_id}", "events"
        )
        self._delete_parent_doc("safety_audit", user_id)

        # 8. Delete subcollection: health_metrics/{user_id}/daily/*
        counts["health_metrics"] = self._delete_subcollection(
            f"health_metrics/{user_id}", "daily"
        )
        self._delete_parent_doc("health_metrics", user_id)

        # 9. Delete chat_sessions WHERE user_id == target (query-based)
        chat_count = 0
        try:
            docs = (
                self.db.collection("chat_sessions")
                .where("user_id", "==", user_id)
                .stream()
            )
            for doc in docs:
                doc.reference.delete()
                chat_count += 1
        except Exception as e:
            logger.warning(f"Deletion: failed to delete chat_sessions: {e}")
        counts["chat_sessions"] = chat_count

        # 10. Delete single-document collections
        for collection_name in [
            "user_profiles",
            "interaction_metrics",
            "bandit_state",
            self.CONSENT_COLLECTION,
        ]:
            try:
                self.db.collection(collection_name).document(user_id).delete()
                counts[collection_name] = 1
            except Exception as e:
                logger.warning(f"Deletion: failed to delete {collection_name}/{user_id}: {e}")
                counts[collection_name] = 0

        # 11. Delete training_examples WHERE user_id == target (Phase 18)
        training_count = 0
        try:
            docs = (
                self.db.collection("training_examples")
                .where("user_id", "==", user_id)
                .stream()
            )
            for doc in docs:
                doc.reference.delete()
                training_count += 1
        except Exception as e:
            logger.warning(f"Deletion: failed to delete training_examples: {e}")
        counts["training_examples"] = training_count

        # 11b. Delete model_performance WHERE user_id == target (Phase 18)
        perf_count = 0
        try:
            docs = (
                self.db.collection("model_performance")
                .where("user_id", "==", user_id)
                .stream()
            )
            for doc in docs:
                doc.reference.delete()
                perf_count += 1
        except Exception as e:
            logger.warning(f"Deletion: failed to delete model_performance: {e}")
        counts["model_performance"] = perf_count

        # 12. Delete escalation_queue WHERE user_id == target (Phase 20)
        esc_count = 0
        try:
            docs = (
                self.db.collection("escalation_queue")
                .where("user_id", "==", user_id)
                .stream()
            )
            for doc in docs:
                doc.reference.delete()
                esc_count += 1
        except Exception as e:
            logger.warning(f"Deletion: failed to delete escalation_queue: {e}")
        counts["escalation_events"] = esc_count

        # 13. Delete subcollection: annual_reviews/{user_id}/reviews/* (Phase 20)
        counts["annual_reviews"] = self._delete_subcollection(
            f"annual_reviews/{user_id}", "reviews"
        )
        self._delete_parent_doc("annual_reviews", user_id)

        # 14. Clear in-memory caches
        try:
            cache = get_cache()
            cache.invalidate(user_profile_key(user_id))
            cache.invalidate(effectiveness_key(user_id))
            cache.invalidate_prefix(f"user:{user_id}")
            cache_cleared = True
        except Exception:
            cache_cleared = False

        total = sum(counts.values())

        # Log deletion completion
        audit_svc.log(
            user_id=user_id,
            action=AuditAction.DELETE,
            resource_type="all_data",
            details=f"Cascade deletion complete: {total} documents deleted",
            actor_id=user_id,
            ip_address=ip_address,
        )

        logger.info(
            f"GDPR erasure complete for {user_id}: "
            f"{total} documents deleted across {len(counts)} collections"
        )

        return DeletionReceipt(
            user_id=user_id,
            collections_deleted=counts,
            total_documents_deleted=total,
            cache_cleared=cache_cleared,
        )

    # ── Consent Management ───────────────────────────────

    def record_consent(
        self,
        user_id: str,
        consents: dict,
        privacy_policy_version: str = "1.0",
        ip_address: str = "",
    ) -> Optional[ConsentRecord]:
        """Record initial user consent."""
        record = ConsentRecord(
            user_id=user_id,
            consents=consents,
            privacy_policy_version=privacy_policy_version,
            ip_address=ip_address,
        )

        if not self.db:
            return record

        try:
            doc_ref = self.db.collection(self.CONSENT_COLLECTION).document(user_id)
            doc_ref.set(record.model_dump())

            # Log consent to audit
            audit_svc = get_audit_service()
            audit_svc.log(
                user_id=user_id,
                action=AuditAction.CONSENT_CHANGE,
                resource_type="consent",
                details=f"Initial consent recorded: v{privacy_policy_version}",
                actor_id=user_id,
                ip_address=ip_address,
            )

            return record
        except Exception as e:
            logger.warning(f"Failed to record consent for {user_id}: {e}")
            return None

    def get_consent(self, user_id: str) -> Optional[ConsentRecord]:
        """Get current consent status for a user."""
        if not self.db:
            return None

        try:
            doc = self.db.collection(self.CONSENT_COLLECTION).document(user_id).get()
            if doc.exists:
                data = doc.to_dict()
                return ConsentRecord(**data)
            return None
        except Exception as e:
            logger.warning(f"Failed to get consent for {user_id}: {e}")
            return None

    def update_consent(
        self,
        user_id: str,
        consents: dict,
        privacy_policy_version: str = "",
        ip_address: str = "",
    ) -> Optional[ConsentRecord]:
        """Update consent preferences. Merges with existing consents."""
        existing = self.get_consent(user_id)
        if not existing:
            return self.record_consent(
                user_id, consents, privacy_policy_version or "1.0", ip_address
            )

        # Merge consents
        merged = existing.consents.copy()
        merged.update(consents)

        updated = ConsentRecord(
            user_id=user_id,
            consents=merged,
            privacy_policy_version=privacy_policy_version or existing.privacy_policy_version,
            consented_at=existing.consented_at,
            updated_at=datetime.now().isoformat(),
            ip_address=ip_address or existing.ip_address,
        )

        if not self.db:
            return updated

        try:
            doc_ref = self.db.collection(self.CONSENT_COLLECTION).document(user_id)
            doc_ref.set(updated.model_dump())

            # Log consent change
            audit_svc = get_audit_service()
            audit_svc.log(
                user_id=user_id,
                action=AuditAction.CONSENT_CHANGE,
                resource_type="consent",
                details=f"Consent updated: {list(consents.keys())}",
                actor_id=user_id,
                ip_address=ip_address,
            )

            return updated
        except Exception as e:
            logger.warning(f"Failed to update consent for {user_id}: {e}")
            return None

    # ── Data Retention ───────────────────────────────────

    def enforce_retention_policies(self) -> dict:
        """
        Scan all data types and delete records older than retention windows.
        Returns a summary of what was purged per collection.
        """
        if not self.db:
            return {"status": "no_database"}

        cfg = self._config
        summary = {}

        # Define retention map: (parent_collection, subcollection, date_field, retention_days)
        retention_targets = [
            ("chat_sessions", None, "created_at", cfg.RETENTION_CHAT_SESSIONS_DAYS),
            ("user_memories", "memories", "created_at", cfg.RETENTION_MEMORIES_DAYS),
            ("feedback", "response_feedback", "created_at", cfg.RETENTION_FEEDBACK_DAYS),
            ("feedback", "exercise_outcomes", "created_at", cfg.RETENTION_FEEDBACK_DAYS),
            ("exercise_sessions", "sessions", "started_at", cfg.RETENTION_EXERCISE_SESSIONS_DAYS),
            ("exercise_sessions", "tasks", "assigned_at", cfg.RETENTION_EXERCISE_SESSIONS_DAYS),
            ("soundscape_sessions", "sessions", "started_at", cfg.RETENTION_SOUNDSCAPE_SESSIONS_DAYS),
            ("safety_audit", "events", "created_at", cfg.RETENTION_SAFETY_EVENTS_DAYS),
        ]

        for parent_col, sub_col, date_field, retention_days in retention_targets:
            cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
            key = f"{parent_col}/{sub_col}" if sub_col else parent_col

            try:
                if sub_col:
                    count = self._purge_subcollection_by_date(
                        parent_col, sub_col, date_field, cutoff
                    )
                else:
                    count = self._purge_flat_collection_by_date(
                        parent_col, date_field, cutoff
                    )
                summary[key] = count
                if count > 0:
                    logger.info(f"Retention purge: {key} -> {count} docs deleted")
            except Exception as e:
                logger.warning(f"Retention purge failed for {key}: {e}")
                summary[key] = f"error: {str(e)}"

        # Also purge old audit logs
        try:
            audit_svc = get_audit_service()
            audit_count = audit_svc.purge_old_logs()
            summary["audit_log"] = audit_count
        except Exception as e:
            summary["audit_log"] = f"error: {str(e)}"

        return summary

    def get_retention_policies(self) -> List[dict]:
        """Return the current retention policy configuration."""
        cfg = self._config
        return [
            {"data_type": "chat_sessions", "retention_days": cfg.RETENTION_CHAT_SESSIONS_DAYS},
            {"data_type": "memories", "retention_days": cfg.RETENTION_MEMORIES_DAYS},
            {"data_type": "feedback", "retention_days": cfg.RETENTION_FEEDBACK_DAYS},
            {"data_type": "exercise_sessions", "retention_days": cfg.RETENTION_EXERCISE_SESSIONS_DAYS},
            {"data_type": "soundscape_sessions", "retention_days": cfg.RETENTION_SOUNDSCAPE_SESSIONS_DAYS},
            {"data_type": "safety_events", "retention_days": cfg.RETENTION_SAFETY_EVENTS_DAYS},
            {"data_type": "interaction_metrics", "retention_days": cfg.RETENTION_INTERACTION_METRICS_DAYS},
            {"data_type": "audit_log", "retention_days": cfg.RETENTION_AUDIT_LOG_DAYS},
        ]

    # ── Private Helpers ──────────────────────────────────

    def _read_subcollection(
        self, parent_path: str, subcollection_name: str
    ) -> List[dict]:
        """Read all documents from a subcollection. Returns list of dicts."""
        if not self.db:
            return []

        try:
            parts = parent_path.split("/")
            ref = self.db.collection(parts[0]).document(parts[1])
            docs = ref.collection(subcollection_name).stream()
            results = []
            for doc in docs:
                data = doc.to_dict()
                data["_doc_id"] = doc.id
                results.append(data)
            return results
        except Exception as e:
            logger.warning(f"Failed to read {parent_path}/{subcollection_name}: {e}")
            return []

    def _delete_subcollection(
        self, parent_path: str, subcollection_name: str, batch_size: int = 400
    ) -> int:
        """
        Delete all documents in a Firestore subcollection.
        Uses batched writes for efficiency.
        Returns count of deleted documents.
        """
        if not self.db:
            return 0

        try:
            parts = parent_path.split("/")
            ref = self.db.collection(parts[0]).document(parts[1])
            docs = ref.collection(subcollection_name).stream()

            count = 0
            batch = self.db.batch()
            for doc in docs:
                batch.delete(doc.reference)
                count += 1
                if count % batch_size == 0:
                    batch.commit()
                    batch = self.db.batch()

            if count % batch_size != 0:
                batch.commit()

            return count
        except Exception as e:
            logger.warning(
                f"Failed to delete subcollection {parent_path}/{subcollection_name}: {e}"
            )
            return 0

    def _delete_parent_doc(self, collection: str, doc_id: str) -> bool:
        """Delete a parent document (after its subcollections are cleared)."""
        if not self.db:
            return False

        try:
            self.db.collection(collection).document(doc_id).delete()
            return True
        except Exception as e:
            logger.warning(f"Failed to delete {collection}/{doc_id}: {e}")
            return False

    def _purge_subcollection_by_date(
        self,
        parent_collection: str,
        subcollection_name: str,
        date_field: str,
        cutoff_iso: str,
    ) -> int:
        """
        Iterate all parent docs in a collection, then for each,
        stream the subcollection and delete docs older than cutoff.
        Returns total count deleted.
        """
        if not self.db:
            return 0

        total = 0
        try:
            parent_docs = self.db.collection(parent_collection).stream()
            for parent_doc in parent_docs:
                sub_docs = (
                    parent_doc.reference.collection(subcollection_name)
                    .where(date_field, "<", cutoff_iso)
                    .stream()
                )
                batch = self.db.batch()
                count = 0
                for doc in sub_docs:
                    batch.delete(doc.reference)
                    count += 1
                    if count % 400 == 0:
                        batch.commit()
                        batch = self.db.batch()
                if count % 400 != 0:
                    batch.commit()
                total += count
        except Exception as e:
            logger.warning(
                f"Retention purge failed for {parent_collection}/{subcollection_name}: {e}"
            )
        return total

    def _purge_flat_collection_by_date(
        self,
        collection_name: str,
        date_field: str,
        cutoff_iso: str,
    ) -> int:
        """Delete documents in a flat collection older than cutoff."""
        if not self.db:
            return 0

        try:
            docs = (
                self.db.collection(collection_name)
                .where(date_field, "<", cutoff_iso)
                .stream()
            )
            count = 0
            batch = self.db.batch()
            for doc in docs:
                batch.delete(doc.reference)
                count += 1
                if count % 400 == 0:
                    batch.commit()
                    batch = self.db.batch()
            if count % 400 != 0:
                batch.commit()
            return count
        except Exception as e:
            logger.warning(f"Retention purge failed for {collection_name}: {e}")
            return 0


# ── Singleton ────────────────────────────────────────────

_compliance_service: Optional[ComplianceService] = None


def get_compliance_service() -> ComplianceService:
    """Get or create ComplianceService singleton."""
    global _compliance_service
    if _compliance_service is None:
        _compliance_service = ComplianceService()
    return _compliance_service
