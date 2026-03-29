"""
LucilleLLM - Monitoring Dashboard Service (Phase 19)

Aggregation service providing admin-level metrics across all users.
Caches results using a dedicated TTLCache instance to avoid querying
Firestore on every dashboard page load.

Firestore collections read (no writes):
    user_profiles, chat_sessions, feedback/{user_id}/exercise_outcomes/,
    exercise_sessions/{user_id}/sessions/, safety_audit/{user_id}/events/,
    bandit_state/{user_id}, model_performance/{record_id},
    interaction_metrics/{user_id}

Follows the singleton pattern from other services.
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from config import get_config
from cache import TTLCache
from models import (
    DashboardSystemMetrics,
    DashboardEngagementMetrics,
    DashboardTherapyMetrics,
    DashboardSafetyMetrics,
    DashboardModelMetrics,
    DashboardRLMetrics,
    DashboardAllMetrics,
)

logger = logging.getLogger(__name__)


class MonitoringService:
    """
    Admin-level aggregation service for the monitoring dashboard.

    Each get_*_metrics() method returns a Pydantic model and is
    cached with a configurable TTL to avoid expensive Firestore
    queries on every request.
    """

    def __init__(self):
        self._config = get_config()
        self._db = None
        self._cache = TTLCache(
            default_ttl=self._config.DASHBOARD_CACHE_TTL,
            max_size=20,
        )

        try:
            from firebase_service import get_firebase_service
            fb = get_firebase_service()
            self._db = fb.db
        except Exception as e:
            logger.warning(f"MonitoringService: Firebase not available -- {e}")

        logger.info(
            f"MonitoringService initialized -- "
            f"cache_ttl={self._config.DASHBOARD_CACHE_TTL}s, "
            f"db_available={self._db is not None}"
        )

    # ── System Health ────────────────────────────────────────

    def get_system_metrics(self) -> DashboardSystemMetrics:
        """
        System health panel: uptime, requests, errors, latency, cache, memory.
        Reads from in-memory MetricsCollector + TTLCache. No Firestore needed.
        """
        cached = self._cache.get("dashboard:system")
        if cached is not None:
            return cached

        try:
            from middleware import get_metrics_collector
            from cache import get_cache

            collector = get_metrics_collector()
            summary = collector.get_summary()
            cache_obj = get_cache()
            cache_stats = cache_obj.stats()
            config = self._config

            # Aggregate across all endpoints
            total_requests = 0
            total_errors = 0
            top_endpoints = []

            for path, data in summary.get("endpoints", {}).items():
                total_requests += data.get("count", 0)
                total_errors += data.get("errors", 0)
                top_endpoints.append({
                    "path": path,
                    "count": data["count"],
                    "error_rate": data.get("error_rate", 0),
                    "latency_p50": data.get("latency_ms", {}).get("p50", 0),
                    "latency_p95": data.get("latency_ms", {}).get("p95", 0),
                })

            # Sort by count descending, take top 10
            top_endpoints.sort(key=lambda x: x["count"], reverse=True)
            top_endpoints = top_endpoints[:10]

            # Weighted-average latency across endpoints
            weighted_latencies = []
            for path, data in summary.get("endpoints", {}).items():
                lat = data.get("latency_ms", {})
                if lat and data["count"] > 0:
                    weighted_latencies.append((
                        lat.get("p50", 0),
                        lat.get("p95", 0),
                        lat.get("p99", 0),
                        data["count"],
                    ))

            total_weight = sum(w[3] for w in weighted_latencies) or 1
            p50 = sum(w[0] * w[3] for w in weighted_latencies) / total_weight
            p95 = sum(w[1] * w[3] for w in weighted_latencies) / total_weight
            p99 = sum(w[2] * w[3] for w in weighted_latencies) / total_weight

            # Memory usage (optional psutil)
            rss_mb = 0.0
            mem_percent = 0.0
            try:
                import psutil
                proc = psutil.Process()
                rss_mb = round(proc.memory_info().rss / 1024 / 1024, 1)
                mem_percent = round(proc.memory_percent(), 1)
            except (ImportError, Exception):
                pass

            result = DashboardSystemMetrics(
                uptime_seconds=summary.get("uptime_seconds", 0),
                total_requests=total_requests,
                total_errors=total_errors,
                overall_error_rate=round(total_errors / max(total_requests, 1), 4),
                latency_p50=round(p50, 1),
                latency_p95=round(p95, 1),
                latency_p99=round(p99, 1),
                cache_hit_rate=cache_stats.get("hit_rate", 0),
                cache_size=cache_stats.get("size", 0),
                memory_rss_mb=rss_mb,
                memory_percent=mem_percent,
                environment=config.ENVIRONMENT,
                model=config.OPENAI_MODEL,
                top_endpoints=top_endpoints,
            )

            self._cache.set("dashboard:system", result, ttl=15)
            return result

        except Exception as e:
            logger.error(f"Failed to compute system metrics: {e}")
            return DashboardSystemMetrics()

    # ── User Engagement ──────────────────────────────────────

    def get_engagement_metrics(self) -> DashboardEngagementMetrics:
        """
        User engagement panel: total users, active today, sessions, messages.
        Queries: user_profiles, interaction_metrics, chat_sessions.
        """
        cached = self._cache.get("dashboard:engagement")
        if cached is not None:
            return cached

        if self._db is None:
            return DashboardEngagementMetrics()

        try:
            # 1. Total users + new users in last 7 days
            user_docs = list(self._db.collection("user_profiles").limit(10000).stream())
            total_users = len(user_docs)

            seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
            new_users_7d = 0
            for doc in user_docs:
                data = doc.to_dict()
                created = data.get("created_at", "")
                # Handle both ISO string and Firestore timestamps
                if hasattr(created, "isoformat"):
                    created = created.isoformat()
                if isinstance(created, str) and created > seven_days_ago:
                    new_users_7d += 1

            # 2. Interaction metrics: active today + messages today
            active_today = 0
            total_messages_today = 0
            metrics_docs = list(
                self._db.collection("interaction_metrics").limit(10000).stream()
            )
            for doc in metrics_docs:
                data = doc.to_dict()
                msgs = data.get("messages_today", 0)
                if msgs > 0:
                    active_today += 1
                    total_messages_today += msgs

            # 3. Total sessions
            session_count = 0
            for _ in self._db.collection("chat_sessions").limit(10000).stream():
                session_count += 1

            result = DashboardEngagementMetrics(
                total_users=total_users,
                active_users_today=active_today,
                total_sessions=session_count,
                total_messages_today=total_messages_today,
                new_users_7d=new_users_7d,
                avg_sessions_per_user=round(
                    session_count / max(total_users, 1), 1
                ),
            )

            self._cache.set("dashboard:engagement", result)
            return result

        except Exception as e:
            logger.error(f"Failed to compute engagement metrics: {e}")
            return DashboardEngagementMetrics()

    # ── Therapy Effectiveness ────────────────────────────────

    def get_therapy_metrics(self) -> DashboardTherapyMetrics:
        """
        Therapy effectiveness panel: exercise outcomes aggregated across users.
        Queries: feedback/{user_id}/exercise_outcomes/, exercise_sessions/.
        Bounded to 200 users, 100 outcomes each.
        """
        cached = self._cache.get("dashboard:therapy")
        if cached is not None:
            return cached

        if self._db is None:
            return DashboardTherapyMetrics()

        try:
            # Get user IDs (limit 200 for cost bounding)
            user_docs = self._db.collection("user_profiles").limit(200).stream()
            user_ids = [doc.id for doc in user_docs]

            total_outcomes = 0
            helpfulness_sum = 0.0
            mood_delta_sum = 0.0
            improved = 0
            unchanged = 0
            worsened = 0
            modality_data = defaultdict(
                lambda: {"count": 0, "helpfulness_sum": 0.0, "mood_delta_sum": 0.0}
            )
            exercise_data = defaultdict(
                lambda: {"count": 0, "helpfulness_sum": 0.0}
            )
            total_started = 0
            total_completed = 0

            for uid in user_ids:
                # Exercise outcomes
                outcomes = list(
                    self._db.collection("feedback")
                    .document(uid)
                    .collection("exercise_outcomes")
                    .limit(100)
                    .stream()
                )
                for doc in outcomes:
                    data = doc.to_dict()
                    total_outcomes += 1
                    h = data.get("helpfulness", 3)
                    helpfulness_sum += h
                    mb = data.get("mood_before", 5)
                    ma = data.get("mood_after", 5)
                    delta = ma - mb
                    mood_delta_sum += delta

                    if delta > 0:
                        improved += 1
                    elif delta == 0:
                        unchanged += 1
                    else:
                        worsened += 1

                    mod = data.get("modality", "unknown")
                    modality_data[mod]["count"] += 1
                    modality_data[mod]["helpfulness_sum"] += h
                    modality_data[mod]["mood_delta_sum"] += delta

                    eid = data.get("exercise_id", "unknown")
                    exercise_data[eid]["count"] += 1
                    exercise_data[eid]["helpfulness_sum"] += h

                # Exercise sessions for completion rate
                sessions = list(
                    self._db.collection("exercise_sessions")
                    .document(uid)
                    .collection("sessions")
                    .limit(100)
                    .stream()
                )
                for doc in sessions:
                    data = doc.to_dict()
                    total_started += 1
                    if data.get("status") == "completed":
                        total_completed += 1

            # Build modality breakdown
            modality_breakdown = {}
            for mod, d in modality_data.items():
                c = d["count"]
                modality_breakdown[mod] = {
                    "count": c,
                    "avg_helpfulness": round(d["helpfulness_sum"] / max(c, 1), 2),
                    "avg_mood_delta": round(d["mood_delta_sum"] / max(c, 1), 2),
                }

            # Top 5 exercises by avg helpfulness (min 2 outcomes)
            top_exercises = []
            for eid, d in exercise_data.items():
                if d["count"] >= 2:
                    top_exercises.append({
                        "exercise_id": eid,
                        "count": d["count"],
                        "avg_helpfulness": round(
                            d["helpfulness_sum"] / d["count"], 2
                        ),
                    })
            top_exercises.sort(
                key=lambda x: x["avg_helpfulness"], reverse=True
            )
            top_exercises = top_exercises[:5]

            result = DashboardTherapyMetrics(
                total_exercise_outcomes=total_outcomes,
                avg_helpfulness=round(
                    helpfulness_sum / max(total_outcomes, 1), 2
                ),
                exercise_completion_rate=round(
                    total_completed / max(total_started, 1), 3
                ),
                avg_mood_improvement=round(
                    mood_delta_sum / max(total_outcomes, 1), 2
                ),
                modality_breakdown=modality_breakdown,
                top_exercises=top_exercises,
                mood_improvement_distribution={
                    "improved": improved,
                    "unchanged": unchanged,
                    "worsened": worsened,
                },
            )

            self._cache.set("dashboard:therapy", result)
            return result

        except Exception as e:
            logger.error(f"Failed to compute therapy metrics: {e}")
            return DashboardTherapyMetrics()

    # ── Safety Overview ──────────────────────────────────────

    def get_safety_metrics(self) -> DashboardSafetyMetrics:
        """
        Safety overview panel: events by risk/type, recent critical events.
        Queries: safety_audit/{user_id}/events/ across users.
        """
        cached = self._cache.get("dashboard:safety")
        if cached is not None:
            return cached

        if self._db is None:
            return DashboardSafetyMetrics()

        try:
            user_docs = self._db.collection("safety_audit").limit(500).stream()
            user_ids = [doc.id for doc in user_docs]

            total = 0
            by_risk = defaultdict(int)
            by_type = defaultdict(int)
            critical_events = []

            for uid in user_ids:
                events = list(
                    self._db.collection("safety_audit")
                    .document(uid)
                    .collection("events")
                    .limit(50)
                    .stream()
                )
                for doc in events:
                    data = doc.to_dict()
                    total += 1
                    risk = data.get("risk_level", "low")
                    etype = data.get("event_type", "unknown")
                    by_risk[risk] += 1
                    by_type[etype] += 1

                    if risk in ("critical", "high", "CRITICAL", "HIGH"):
                        snippet = data.get("message_snippet", "")
                        critical_events.append({
                            "event_id": data.get("event_id", doc.id),
                            "risk_level": risk,
                            "event_type": etype,
                            "action_taken": data.get("action_taken", ""),
                            "created_at": data.get("created_at", ""),
                            "snippet": (snippet[:50] + "...")
                                       if len(snippet) > 50 else snippet,
                        })

            # Sort critical events by created_at descending, take 10
            critical_events.sort(
                key=lambda x: x.get("created_at", ""), reverse=True
            )
            critical_events = critical_events[:10]

            result = DashboardSafetyMetrics(
                total_events=total,
                events_by_risk_level=dict(by_risk),
                events_by_type=dict(by_type),
                recent_critical_events=critical_events,
            )

            self._cache.set("dashboard:safety", result)
            return result

        except Exception as e:
            logger.error(f"Failed to compute safety metrics: {e}")
            return DashboardSafetyMetrics()

    # ── Model A/B Performance ────────────────────────────────

    def get_model_metrics(self) -> DashboardModelMetrics:
        """
        Model A/B panel: reuses FineTuningService.compute_ab_stats().
        """
        cached = self._cache.get("dashboard:models")
        if cached is not None:
            return cached

        try:
            from finetuning_service import get_finetuning_service
            ft_svc = get_finetuning_service()
            stats = ft_svc.compute_ab_stats()
            config = self._config

            result = DashboardModelMetrics(
                ft_enabled=config.FT_ENABLED,
                active_model_id=stats.active_model_id,
                ab_split_percent=stats.ab_split_percent,
                base_total=stats.total_responses_base,
                base_helpful_rate=stats.helpful_rate_base,
                base_avg_length=stats.avg_response_length_base,
                ft_total=stats.total_responses_ft,
                ft_helpful_rate=stats.helpful_rate_ft,
                ft_avg_length=stats.avg_response_length_ft,
            )

            self._cache.set("dashboard:models", result)
            return result

        except Exception as e:
            logger.error(f"Failed to compute model metrics: {e}")
            return DashboardModelMetrics(
                ft_enabled=self._config.FT_ENABLED,
            )

    # ── RL Bandit Overview ───────────────────────────────────

    def get_rl_metrics(self) -> DashboardRLMetrics:
        """
        RL bandit overview panel: aggregate arm stats across all users.
        Queries: bandit_state/{user_id} for all users.
        """
        cached = self._cache.get("dashboard:rl")
        if cached is not None:
            return cached

        config = self._config
        if self._db is None:
            return DashboardRLMetrics(rl_enabled=config.RL_ENABLED)

        if not config.RL_ENABLED:
            return DashboardRLMetrics(rl_enabled=False)

        try:
            docs = list(
                self._db.collection("bandit_state").limit(500).stream()
            )
            total_users = len(docs)
            total_arms = 0

            # Aggregate: group -> exercise_id -> {alpha_sum, beta_sum, count}
            group_agg = defaultdict(
                lambda: defaultdict(
                    lambda: {"alpha_sum": 0.0, "beta_sum": 0.0, "count": 0}
                )
            )

            for doc in docs:
                data = doc.to_dict()
                arms = data.get("arms", {})
                total_arms += len(arms)

                for arm_key, arm_data in arms.items():
                    try:
                        inner = arm_key.strip("()")
                        group, exercise_id = inner.split(",", 1)
                        alpha = arm_data.get("alpha", 1.0)
                        beta_val = arm_data.get("beta", 1.0)

                        agg = group_agg[group][exercise_id]
                        agg["alpha_sum"] += alpha
                        agg["beta_sum"] += beta_val
                        agg["count"] += 1
                    except (ValueError, AttributeError):
                        continue

            # Top 3 arms per group by mean success rate
            top_arms_by_group = {}
            for group, exercises in group_agg.items():
                arms_list = []
                for eid, agg in exercises.items():
                    total_alpha = agg["alpha_sum"]
                    total_beta = agg["beta_sum"]
                    mean = (
                        total_alpha / (total_alpha + total_beta)
                        if (total_alpha + total_beta) > 0
                        else 0.5
                    )
                    observations = agg["count"]
                    arms_list.append({
                        "exercise_id": eid,
                        "mean_success_rate": round(mean, 3),
                        "total_users": observations,
                        "avg_alpha": round(
                            total_alpha / max(observations, 1), 2
                        ),
                        "avg_beta": round(
                            total_beta / max(observations, 1), 2
                        ),
                    })
                arms_list.sort(
                    key=lambda x: x["mean_success_rate"], reverse=True
                )
                top_arms_by_group[group] = arms_list[:3]

            result = DashboardRLMetrics(
                rl_enabled=True,
                total_users_with_bandit_state=total_users,
                total_arms=total_arms,
                top_arms_by_group=top_arms_by_group,
            )

            self._cache.set("dashboard:rl", result)
            return result

        except Exception as e:
            logger.error(f"Failed to compute RL metrics: {e}")
            return DashboardRLMetrics(rl_enabled=config.RL_ENABLED)

    # ── Combined ─────────────────────────────────────────────

    def get_all_metrics(self) -> DashboardAllMetrics:
        """Get all dashboard sections in one call."""
        return DashboardAllMetrics(
            system=self.get_system_metrics(),
            engagement=self.get_engagement_metrics(),
            therapy=self.get_therapy_metrics(),
            safety=self.get_safety_metrics(),
            models=self.get_model_metrics(),
            rl=self.get_rl_metrics(),
        )


# ── Singleton ─────────────────────────────────────────────

_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """Get or create MonitoringService singleton."""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service
