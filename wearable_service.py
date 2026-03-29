"""
LucilleLLM - Wearable Integration Service

Provides health data ingestion, storage, analytics, and prompt formatting
for wearable/health metrics (sleep + activity).

Architecture: Provider-agnostic. The backend receives structured JSON metrics
from the Flutter client, which handles all device SDK integration (Apple
HealthKit, Google Health Connect). No paid APIs needed on the backend.

Firestore structure: health_metrics/{user_id}/daily/{YYYY-MM-DD}

Follows the singleton pattern from other services.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from config import get_config
from models import (
    ActivityRecord,
    DailyHealthMetrics,
    HealthSummary,
    HealthSyncRequest,
    SleepRecord,
)

logger = logging.getLogger(__name__)


class WearableService:
    """
    Service for wearable health data operations.

    Data flow:
      Flutter client -> POST /wearables/{user_id}/sync -> Firestore
      Firestore -> format_health_context() -> LLM system prompt
    """

    def __init__(self):
        self._config = get_config()
        self._db = None

        try:
            from firebase_service import get_firebase_service
            fb = get_firebase_service()
            self._db = fb.db
        except Exception as e:
            logger.warning(f"WearableService: Firebase not available — {e}")

        logger.info(
            f"WearableService initialized — "
            f"sync_enabled={self._config.WEARABLE_SYNC_ENABLED}, "
            f"context_days={self._config.WEARABLE_HEALTH_CONTEXT_DAYS}, "
            f"db_available={self._db is not None}"
        )

    # ── Properties ────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Check if wearable service has database access."""
        return self._db is not None

    # ── Data Ingestion ────────────────────────────────────────

    def sync_health_data(
        self, request: HealthSyncRequest
    ) -> Tuple[int, Optional[str]]:
        """
        Batch upsert daily health metrics to Firestore.

        Uses doc.set(data, merge=True) for idempotent upserts — safe to
        call multiple times with the same data.

        Returns:
            Tuple of (records_saved, error_message).
        """
        if not self._db:
            return 0, "Database not available"

        if not self._config.WEARABLE_SYNC_ENABLED:
            return 0, "Wearable sync is disabled"

        if not request.metrics:
            return 0, "No metrics provided"

        saved = 0
        for metric in request.metrics:
            try:
                data = metric.model_dump(exclude_none=True)
                # Override source from request-level if metric doesn't specify
                if metric.source == "manual" and request.source != "manual":
                    data["source"] = request.source
                # Stamp sync time
                data["synced_at"] = datetime.now().isoformat()

                doc_ref = (
                    self._db.collection("health_metrics")
                    .document(request.user_id)
                    .collection("daily")
                    .document(metric.date)
                )
                doc_ref.set(data, merge=True)
                saved += 1
            except Exception as e:
                logger.error(
                    f"Failed to save metric for {request.user_id}/{metric.date}: {e}"
                )

        logger.info(
            f"Health sync: saved {saved}/{len(request.metrics)} records "
            f"for user {request.user_id}"
        )
        return saved, None

    # ── Data Retrieval ────────────────────────────────────────

    def get_daily_metrics(
        self, user_id: str, date: str
    ) -> Optional[dict]:
        """Get health metrics for a single day."""
        if not self._db:
            return None

        try:
            doc = (
                self._db.collection("health_metrics")
                .document(user_id)
                .collection("daily")
                .document(date)
                .get()
            )
            if doc.exists:
                data = doc.to_dict()
                data["date"] = doc.id
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to get daily metrics: {e}")
            return None

    def get_recent_metrics(
        self, user_id: str, days: int = 7
    ) -> List[dict]:
        """
        Get last N days of health metrics, ordered by date descending.
        """
        if not self._db:
            return []

        try:
            # Calculate date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            docs = (
                self._db.collection("health_metrics")
                .document(user_id)
                .collection("daily")
                .where("date", ">=", start_date)
                .where("date", "<=", end_date)
                .order_by("date", direction="DESCENDING")
                .stream()
            )

            results = []
            for doc in docs:
                data = doc.to_dict()
                data["date"] = doc.id
                results.append(data)

            return results
        except Exception as e:
            logger.error(f"Failed to get recent metrics: {e}")
            return []

    # ── Analytics ─────────────────────────────────────────────

    def get_health_summary(
        self, user_id: str, days: int = 7
    ) -> HealthSummary:
        """
        Get aggregated health summary with trends.

        Trend calculation: compare last days/2 vs previous days/2.
        >10% change = improving/declining, otherwise stable.
        """
        metrics = self.get_recent_metrics(user_id, days=days)

        if not metrics:
            return HealthSummary(
                user_id=user_id,
                period_days=days,
                total_records=0,
                status="no_data",
                timestamp=datetime.now().isoformat(),
            )

        # Extract sleep and activity values
        sleep_hours = []
        steps_list = []
        active_mins = []

        for m in metrics:
            sleep = m.get("sleep")
            if sleep and sleep.get("duration_hours", 0) > 0:
                sleep_hours.append(sleep["duration_hours"])

            activity = m.get("activity")
            if activity:
                if activity.get("steps", 0) > 0:
                    steps_list.append(activity["steps"])
                if activity.get("active_minutes", 0) > 0:
                    active_mins.append(activity["active_minutes"])

        # Calculate averages
        avg_sleep = round(sum(sleep_hours) / len(sleep_hours), 1) if sleep_hours else 0.0
        avg_steps = int(sum(steps_list) / len(steps_list)) if steps_list else 0
        avg_active = int(sum(active_mins) / len(active_mins)) if active_mins else 0

        # Calculate trends (split data into two halves)
        sleep_trend = self._calculate_trend(sleep_hours)
        activity_trend = self._calculate_trend(steps_list)

        return HealthSummary(
            user_id=user_id,
            period_days=days,
            avg_sleep_hours=avg_sleep,
            avg_steps=avg_steps,
            avg_active_minutes=avg_active,
            sleep_trend=sleep_trend,
            activity_trend=activity_trend,
            total_records=len(metrics),
            status="success",
            timestamp=datetime.now().isoformat(),
        )

    def get_sleep_insights(
        self, user_id: str, days: int = 7
    ) -> dict:
        """
        Get sleep-specific analysis.

        Returns dict with avg duration, quality distribution,
        worst/best nights, and recommendations.
        """
        metrics = self.get_recent_metrics(user_id, days=days)

        if not metrics:
            return {
                "status": "no_data",
                "message": "No sleep data available for analysis.",
            }

        sleep_data = []
        quality_counts = {"poor": 0, "fair": 0, "good": 0, "excellent": 0, "unknown": 0}

        for m in metrics:
            sleep = m.get("sleep")
            if sleep and sleep.get("duration_hours", 0) > 0:
                sleep_data.append({
                    "date": m.get("date", "unknown"),
                    "duration_hours": sleep["duration_hours"],
                    "quality": sleep.get("quality", "unknown"),
                    "deep_sleep_minutes": sleep.get("deep_sleep_minutes", 0),
                    "rem_sleep_minutes": sleep.get("rem_sleep_minutes", 0),
                    "awakenings": sleep.get("awakenings", 0),
                })
                quality = sleep.get("quality", "unknown")
                if quality in quality_counts:
                    quality_counts[quality] += 1

        if not sleep_data:
            return {
                "status": "no_data",
                "message": "No sleep records found in the specified period.",
            }

        durations = [s["duration_hours"] for s in sleep_data]
        avg_duration = round(sum(durations) / len(durations), 1)

        # Find best and worst nights
        best_night = max(sleep_data, key=lambda s: s["duration_hours"])
        worst_night = min(sleep_data, key=lambda s: s["duration_hours"])

        # Predominant quality
        non_unknown = {k: v for k, v in quality_counts.items() if k != "unknown" and v > 0}
        predominant_quality = max(non_unknown, key=non_unknown.get) if non_unknown else "unknown"

        return {
            "status": "success",
            "period_days": days,
            "total_nights": len(sleep_data),
            "avg_duration_hours": avg_duration,
            "predominant_quality": predominant_quality,
            "quality_distribution": {k: v for k, v in quality_counts.items() if v > 0},
            "best_night": {"date": best_night["date"], "hours": best_night["duration_hours"]},
            "worst_night": {"date": worst_night["date"], "hours": worst_night["duration_hours"]},
            "avg_deep_sleep_minutes": round(
                sum(s["deep_sleep_minutes"] for s in sleep_data) / len(sleep_data)
            ),
            "avg_rem_sleep_minutes": round(
                sum(s["rem_sleep_minutes"] for s in sleep_data) / len(sleep_data)
            ),
            "avg_awakenings": round(
                sum(s["awakenings"] for s in sleep_data) / len(sleep_data), 1
            ),
        }

    # ── Prompt Formatting ─────────────────────────────────────

    def format_health_context(self, user_id: str) -> str:
        """
        Build health context string for LLM system prompt injection.

        Returns 4-6 line summary of recent health data, or "" if no data.
        Uses WEARABLE_HEALTH_CONTEXT_DAYS from config.
        """
        days = self._config.WEARABLE_HEALTH_CONTEXT_DAYS
        summary = self.get_health_summary(user_id, days=days)

        if summary.total_records == 0:
            return ""

        lines = [f"[Health Context \u2014 Last {days} Days]"]

        # Sleep line
        if summary.avg_sleep_hours > 0:
            trend_icon = self._trend_icon(summary.sleep_trend)
            lines.append(
                f"Sleep: avg {summary.avg_sleep_hours}h/night "
                f"({summary.sleep_trend} {trend_icon})"
            )

        # Activity line
        if summary.avg_steps > 0 or summary.avg_active_minutes > 0:
            trend_icon = self._trend_icon(summary.activity_trend)
            parts = []
            if summary.avg_steps > 0:
                parts.append(f"{summary.avg_steps:,} steps/day")
            if summary.avg_active_minutes > 0:
                parts.append(f"{summary.avg_active_minutes} active min/day")
            lines.append(
                f"Activity: avg {', '.join(parts)} "
                f"({summary.activity_trend} {trend_icon})"
            )

        # Add note for concerning trends
        if summary.sleep_trend == "declining":
            lines.append("Note: Sleep duration has been declining recently.")
        if summary.activity_trend == "declining":
            lines.append("Note: Physical activity has been declining recently.")

        return "\n".join(lines)

    # ── Internal Helpers ──────────────────────────────────────

    @staticmethod
    def _calculate_trend(values: List[float]) -> str:
        """
        Calculate trend by comparing two halves of the data.

        Returns "improving", "declining", or "stable".
        >10% change triggers improving/declining.
        """
        if len(values) < 4:
            return "stable"

        mid = len(values) // 2
        # values are ordered most-recent-first, so:
        # recent = first half, older = second half
        recent_avg = sum(values[:mid]) / mid
        older_avg = sum(values[mid:]) / (len(values) - mid)

        if older_avg == 0:
            return "stable"

        change_ratio = (recent_avg - older_avg) / older_avg

        if change_ratio > 0.10:
            return "improving"
        elif change_ratio < -0.10:
            return "declining"
        return "stable"

    @staticmethod
    def _trend_icon(trend: str) -> str:
        """Get unicode icon for trend direction."""
        return {
            "improving": "\u2191",
            "declining": "\u2193",
            "stable": "\u2192",
        }.get(trend, "\u2192")


# ── Singleton ─────────────────────────────────────────────

_wearable_service: Optional[WearableService] = None


def get_wearable_service() -> WearableService:
    """Get or create WearableService singleton."""
    global _wearable_service
    if _wearable_service is None:
        _wearable_service = WearableService()
    return _wearable_service
