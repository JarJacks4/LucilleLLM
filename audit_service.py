"""
LucilleLLM - Audit Logging Service

Immutable append-only audit log for GDPR/HIPAA compliance.
Logs all data access operations (read, write, delete, export).

Firestore structure:
    audit_log/{auto_id}  (flat collection, auto-generated document IDs)
    NOT keyed to user_id -- audit logs survive user data deletion.

Follows the singleton pattern from other services.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from config import get_config
from firebase_service import get_firebase_service
from models import AuditLogEntry, AuditAction

logger = logging.getLogger(__name__)


class AuditService:
    """
    Immutable audit logging service.

    Key properties:
    - Append-only: no update operations
    - NOT keyed to user_id: survives user data deletion (legal requirement)
    - Logs: who accessed, what data, when, action type, IP address
    - Purge only for entries older than HIPAA retention (7 years default)
    """

    COLLECTION = "audit_log"

    def __init__(self):
        self._firebase = get_firebase_service()
        self._config = get_config()

    @property
    def db(self):
        return self._firebase.db if self._firebase else None

    def log(
        self,
        user_id: str,
        action: AuditAction,
        resource_type: str,
        resource_id: str = "",
        details: str = "",
        actor_id: str = "system",
        ip_address: str = "",
    ) -> Optional[str]:
        """
        Log a data access event. Returns log_id or None on failure.

        This is the primary method called by other services and middleware
        to record data access.
        """
        if not self._config.AUDIT_LOG_ENABLED:
            return None

        entry = AuditLogEntry(
            user_id=user_id,
            actor_id=actor_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
        )

        if not self.db:
            logger.debug(f"Audit log (no db): {entry.action.value} {entry.resource_type}")
            return entry.log_id

        try:
            doc_ref = self.db.collection(self.COLLECTION).document(entry.log_id)
            doc_ref.set(entry.model_dump())
            return entry.log_id
        except Exception as e:
            logger.warning(f"Failed to write audit log: {e}")
            return None

    def query_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        limit: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[dict]:
        """
        Query audit logs with optional filters.
        Returns list of audit log entries as dicts.
        """
        if not self.db:
            return []

        try:
            query = self.db.collection(self.COLLECTION)

            if user_id:
                query = query.where("user_id", "==", user_id)
            if action:
                query = query.where("action", "==", action)
            if resource_type:
                query = query.where("resource_type", "==", resource_type)
            if start_date:
                query = query.where("timestamp", ">=", start_date)
            if end_date:
                query = query.where("timestamp", "<=", end_date)

            query = query.order_by("timestamp", direction="DESCENDING")
            query = query.limit(limit)

            docs = query.stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.warning(f"Audit log query failed: {e}")
            return []

    def purge_old_logs(self) -> int:
        """
        Delete audit logs older than RETENTION_AUDIT_LOG_DAYS.
        Returns count of deleted entries.
        Per HIPAA, default retention is 7 years.
        """
        if not self.db:
            return 0

        cutoff = datetime.now() - timedelta(
            days=self._config.RETENTION_AUDIT_LOG_DAYS
        )
        cutoff_iso = cutoff.isoformat()

        try:
            docs = (
                self.db.collection(self.COLLECTION)
                .where("timestamp", "<", cutoff_iso)
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

            if count > 0:
                logger.info(f"Audit log purge: deleted {count} entries older than {cutoff_iso}")
            return count
        except Exception as e:
            logger.warning(f"Audit log purge failed: {e}")
            return 0


# ── Singleton ────────────────────────────────────────────

_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """Get or create AuditService singleton."""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service
