"""
LucilleLLM - Middleware

Provides rate limiting (token bucket) and request metrics middleware
for the FastAPI application.

Rate limiting protects /chat endpoints from abuse.
Metrics middleware tracks per-endpoint latency and request counts.
"""

import contextvars
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Protocol

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import get_config
from audit_service import get_audit_service
from models import AuditAction

logger = logging.getLogger(__name__)


# ── Request correlation ID context ────────────────────────
# A ContextVar carries the current request's ID through async call stacks so
# any log line, anywhere in the codebase, can be tagged with it without
# threading the value through every function call. The StructuredFormatter
# in main.py reads this and includes it in the JSON log output.
_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def get_request_id() -> str:
    """Return the current request's correlation ID, or empty string if outside a request."""
    return _request_id_ctx.get()


def set_request_id(request_id: str) -> contextvars.Token:
    """Set the correlation ID for the current async context. Returns a token for reset."""
    return _request_id_ctx.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    """Reset the correlation ID context to its previous value."""
    _request_id_ctx.reset(token)

# Regex to normalize UUID segments in paths for metrics grouping
_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


# ── Token Bucket ──────────────────────────────────────────


class TokenBucket:
    """
    Thread-safe token bucket rate limiter.

    Tokens refill at a constant rate up to a maximum capacity.
    Each request consumes one token.
    """

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens added per second
            capacity: Maximum burst size (bucket capacity)
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Try to consume one token. Returns True if successful."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._capacity, self._tokens + elapsed * self._rate
            )
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


# ── Rate Limiter Backends ─────────────────────────────────
#
# Two backends with the same .check(client_id) -> bool interface:
#   - InMemoryRateLimiter: token buckets in process memory. Simple, zero deps,
#     but each Cloud Run instance has its own buckets, so a user hitting 3
#     instances effectively gets 3x the limit. Fine for development and small
#     production deployments.
#   - RedisRateLimiter: sliding window counters in Redis. Shared across all
#     instances — true global rate limiting. Activates automatically if the
#     REDIS_URL env var is set AND the redis package is installed.
#
# To enable Redis: `pip install redis` and set REDIS_URL=redis://host:6379/0


class RateLimiterBackend(Protocol):
    """Interface every rate limiter implementation must satisfy."""
    def check(self, client_id: str) -> bool: ...


class InMemoryRateLimiter:
    """
    Manages global and per-client token buckets in process memory.

    Limitation: state is per-process, so does NOT enforce limits correctly
    across multiple Cloud Run instances. Use RedisRateLimiter for production
    deployments with >1 instance.
    """

    def __init__(
        self,
        chat_rate: int = 10,
        global_rate: int = 100,
        window_seconds: int = 60,
    ):
        self._global_bucket = TokenBucket(
            rate=global_rate / window_seconds,
            capacity=global_rate,
        )
        self._client_buckets: Dict[str, TokenBucket] = {}
        self._chat_rate = chat_rate
        self._window_seconds = window_seconds
        self._lock = threading.Lock()

    def check(self, client_id: str) -> bool:
        """Check if the request is allowed. Returns True if allowed."""
        if not self._global_bucket.consume():
            return False
        with self._lock:
            if client_id not in self._client_buckets:
                self._client_buckets[client_id] = TokenBucket(
                    rate=self._chat_rate / self._window_seconds,
                    capacity=self._chat_rate,
                )
        return self._client_buckets[client_id].consume()


class RedisRateLimiter:
    """
    Distributed sliding-window rate limiter backed by Redis.

    Uses INCR + EXPIRE on a per-(client_id, window) key. Cheap (one round-trip
    per request) and shared across all backend instances, so users can't bypass
    limits by hitting different Cloud Run replicas.

    Falls back gracefully if Redis becomes unreachable: returns True (allow)
    rather than blocking all traffic on a Redis outage. The fallback is logged
    so you'll see it in your monitoring.
    """

    def __init__(
        self,
        redis_url: str,
        chat_rate: int = 10,
        global_rate: int = 100,
        window_seconds: int = 60,
    ):
        import redis  # local import — only required when this backend is used
        self._redis = redis.from_url(redis_url, socket_timeout=2, decode_responses=True)
        self._chat_rate = chat_rate
        self._global_rate = global_rate
        self._window_seconds = window_seconds
        # Verify connectivity at startup so a misconfigured URL fails loudly
        self._redis.ping()

    def check(self, client_id: str) -> bool:
        """Check both global and per-client limits via Redis counters."""
        try:
            now_window = int(time.time() // self._window_seconds)
            global_key = f"rl:global:{now_window}"
            client_key = f"rl:client:{client_id}:{now_window}"

            pipe = self._redis.pipeline()
            pipe.incr(global_key)
            pipe.expire(global_key, self._window_seconds * 2)
            pipe.incr(client_key)
            pipe.expire(client_key, self._window_seconds * 2)
            global_count, _, client_count, _ = pipe.execute()

            if int(global_count) > self._global_rate:
                return False
            if int(client_count) > self._chat_rate:
                return False
            return True
        except Exception as e:
            # Fail-open on Redis errors so we don't block all traffic during
            # an outage. Logged so you can alert on it.
            logger.error(f"Redis rate limiter failed, allowing request: {e}")
            return True


def _build_rate_limiter() -> RateLimiterBackend:
    """Construct the appropriate backend based on REDIS_URL env var."""
    config = get_config()
    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        try:
            limiter = RedisRateLimiter(
                redis_url=redis_url,
                chat_rate=config.RATE_LIMIT_CHAT,
                global_rate=config.RATE_LIMIT_GLOBAL,
                window_seconds=config.RATE_LIMIT_WINDOW_SECONDS,
            )
            logger.info(f"Rate limiter: Redis backend at {redis_url.split('@')[-1]}")
            return limiter
        except Exception as e:
            logger.error(
                f"Redis rate limiter init failed ({e}); falling back to in-memory. "
                f"Multi-instance deployments will NOT enforce limits correctly."
            )
    logger.info("Rate limiter: in-memory backend (single-instance only)")
    return InMemoryRateLimiter(
        chat_rate=config.RATE_LIMIT_CHAT,
        global_rate=config.RATE_LIMIT_GLOBAL,
        window_seconds=config.RATE_LIMIT_WINDOW_SECONDS,
    )


def _get_client_id_from_request(request: Request) -> str:
    """Extract client identifier from request (Cloud Run X-Forwarded-For aware)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


# Backwards-compat alias for existing imports / external code
RateLimiter = InMemoryRateLimiter


# ── Rate Limit Middleware ─────────────────────────────────

# Paths that are rate-limited
_RATE_LIMITED_PATHS = {"/chat", "/chat/stream"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that applies rate limiting to /chat endpoints."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path.rstrip("/")

        if path in _RATE_LIMITED_PATHS:
            limiter = get_rate_limiter()
            client_id = _get_client_id_from_request(request)

            if not limiter.check(client_id):
                logger.warning(f"Rate limit exceeded for client {client_id} on {path}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded. Please try again later.",
                        "retry_after": get_config().RATE_LIMIT_WINDOW_SECONDS,
                    },
                    headers={
                        "Retry-After": str(get_config().RATE_LIMIT_WINDOW_SECONDS)
                    },
                )

        return await call_next(request)


# ── Metrics Collector ─────────────────────────────────────

# Maximum latency entries stored per endpoint to bound memory
_MAX_LATENCY_ENTRIES = 1000


class MetricsCollector:
    """
    Collects per-endpoint request metrics: count, errors, latencies.
    Provides percentile summaries (p50, p95, p99).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._endpoints: Dict[str, dict] = defaultdict(
            lambda: {"count": 0, "errors": 0, "latencies": []}
        )
        self._start_time = time.time()

    def record(self, endpoint: str, status_code: int, latency_ms: float) -> None:
        """Record a request metric."""
        normalized = self._normalize_path(endpoint)
        with self._lock:
            entry = self._endpoints[normalized]
            entry["count"] += 1
            if status_code >= 400:
                entry["errors"] += 1
            latencies = entry["latencies"]
            latencies.append(latency_ms)
            # Bound the latency array
            if len(latencies) > _MAX_LATENCY_ENTRIES:
                entry["latencies"] = latencies[-_MAX_LATENCY_ENTRIES:]

    def get_summary(self) -> dict:
        """Get a summary of all endpoint metrics."""
        with self._lock:
            endpoints = {}
            for path, data in self._endpoints.items():
                latencies = sorted(data["latencies"]) if data["latencies"] else []
                endpoints[path] = {
                    "count": data["count"],
                    "errors": data["errors"],
                    "error_rate": round(
                        data["errors"] / max(data["count"], 1), 3
                    ),
                    "latency_ms": self._percentiles(latencies)
                    if latencies
                    else {},
                }
            return {
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "endpoints": endpoints,
            }

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Replace UUID segments with :id for grouping."""
        return _UUID_PATTERN.sub(":id", path)

    @staticmethod
    def _percentiles(sorted_latencies: List[float]) -> dict:
        """Calculate p50, p95, p99 from sorted latencies."""
        n = len(sorted_latencies)
        if n == 0:
            return {}
        return {
            "p50": round(sorted_latencies[int(n * 0.5)], 1),
            "p95": round(sorted_latencies[min(int(n * 0.95), n - 1)], 1),
            "p99": round(sorted_latencies[min(int(n * 0.99), n - 1)], 1),
            "avg": round(sum(sorted_latencies) / n, 1),
        }


# ── Metrics Middleware ────────────────────────────────────


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware that tracks request latency and adds diagnostic headers.

    Adds:
        X-Request-ID: unique request identifier
        X-Response-Time-Ms: response time in milliseconds
    """

    async def dispatch(self, request: Request, call_next):
        # Honor an upstream X-Request-ID (from load balancer / API gateway) if
        # present, otherwise generate a fresh one. Setting it on the contextvar
        # lets every log line in this request automatically include the ID.
        incoming_id = request.headers.get("x-request-id", "").strip()
        request_id = incoming_id or str(uuid.uuid4())
        token = set_request_id(request_id)
        start = time.time()

        try:
            response = await call_next(request)
        finally:
            reset_request_id(token)

        elapsed_ms = (time.time() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(round(elapsed_ms, 1))

        # Record metrics
        collector = get_metrics_collector()
        collector.record(
            endpoint=request.url.path,
            status_code=response.status_code,
            latency_ms=elapsed_ms,
        )

        return response


# ── Privacy Middleware ────────────────────────────────────

# Regex to extract user_id from URL paths like /users/{user_id}/...
_USER_PATH_PATTERN = re.compile(r"/users/([^/]+)")

# Privacy-related security headers
_PRIVACY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}


class PrivacyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds security headers and logs user-scoped data access
    to the audit trail.

    - Adds privacy/security headers to every response
    - Fires audit log entries for user-scoped endpoints (fire-and-forget)
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add privacy/security headers to every response
        for header, value in _PRIVACY_HEADERS.items():
            response.headers[header] = value

        # Fire-and-forget: log user-scoped data access to audit trail
        try:
            path = request.url.path
            match = _USER_PATH_PATTERN.search(path)
            if match:
                user_id = match.group(1)
                # Skip logging for common non-data endpoints
                if user_id not in ("onboard",):
                    method = request.method
                    action_map = {
                        "GET": AuditAction.READ,
                        "POST": AuditAction.WRITE,
                        "PUT": AuditAction.UPDATE,
                        "DELETE": AuditAction.DELETE,
                    }
                    action = action_map.get(method, AuditAction.READ)
                    # Extract IP address
                    ip = ""
                    forwarded = request.headers.get("x-forwarded-for")
                    if forwarded:
                        ip = forwarded.split(",")[0].strip()
                    elif request.client:
                        ip = request.client.host
                    audit_svc = get_audit_service()
                    audit_svc.log(
                        user_id=user_id,
                        action=action,
                        resource_type="api_endpoint",
                        resource_id=path,
                        details=f"{method} {path} -> {response.status_code}",
                        actor_id=user_id,
                        ip_address=ip,
                    )
        except Exception as e:
            logger.debug(f"Audit trail logging skipped: {e}")

        return response


# ── Singletons ────────────────────────────────────────────

_rate_limiter: Optional[RateLimiterBackend] = None
_metrics_collector: Optional[MetricsCollector] = None


def get_rate_limiter() -> RateLimiterBackend:
    """Get or create the rate limiter singleton (Redis if configured, else in-memory)."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = _build_rate_limiter()
    return _rate_limiter


def get_metrics_collector() -> MetricsCollector:
    """Get or create MetricsCollector singleton."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
