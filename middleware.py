"""
LucilleLLM - Middleware

Provides rate limiting (token bucket) and request metrics middleware
for the FastAPI application.

Rate limiting protects /chat endpoints from abuse.
Metrics middleware tracks per-endpoint latency and request counts.
"""

import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import get_config
from audit_service import get_audit_service
from models import AuditAction

logger = logging.getLogger(__name__)

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


# ── Rate Limiter ──────────────────────────────────────────


class RateLimiter:
    """
    Manages global and per-client token buckets.
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
        # Global check first
        if not self._global_bucket.consume():
            return False

        # Per-client check
        with self._lock:
            if client_id not in self._client_buckets:
                self._client_buckets[client_id] = TokenBucket(
                    rate=self._chat_rate / self._window_seconds,
                    capacity=self._chat_rate,
                )
        return self._client_buckets[client_id].consume()

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Cloud Run sets X-Forwarded-For
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"


# ── Rate Limit Middleware ─────────────────────────────────

# Paths that are rate-limited
_RATE_LIMITED_PATHS = {"/chat", "/chat/stream"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that applies rate limiting to /chat endpoints."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path.rstrip("/")

        if path in _RATE_LIMITED_PATHS:
            limiter = get_rate_limiter()
            client_id = limiter._get_client_id(request)

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
        request_id = str(uuid.uuid4())
        start = time.time()

        response = await call_next(request)

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

_rate_limiter: Optional[RateLimiter] = None
_metrics_collector: Optional[MetricsCollector] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create RateLimiter singleton."""
    global _rate_limiter
    if _rate_limiter is None:
        config = get_config()
        _rate_limiter = RateLimiter(
            chat_rate=config.RATE_LIMIT_CHAT,
            global_rate=config.RATE_LIMIT_GLOBAL,
            window_seconds=config.RATE_LIMIT_WINDOW_SECONDS,
        )
    return _rate_limiter


def get_metrics_collector() -> MetricsCollector:
    """Get or create MetricsCollector singleton."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
