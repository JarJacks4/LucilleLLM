"""
LucilleLLM - In-Memory TTL Cache

Thread-safe in-memory cache with TTL expiration and bounded size.
No external dependencies (no Redis required).
Suitable for Cloud Run where instances are ephemeral.

Follows the singleton pattern from other services.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with expiration."""

    value: Any
    expires_at: float
    created_at: float = field(default_factory=time.time)


class TTLCache:
    """
    Thread-safe in-memory cache with TTL and bounded size.

    - Entries expire after their TTL
    - When max_size is reached, expired entries are evicted first,
      then the oldest entries are removed
    - Tracks hit/miss/eviction stats
    """

    def __init__(self, default_ttl: int = 60, max_size: int = 500):
        self._store: Dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache. Returns None if expired or missing."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.time() > entry.expires_at:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache with optional custom TTL."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()

        with self._lock:
            # If key already exists, just update it
            if key in self._store:
                self._store[key] = CacheEntry(
                    value=value,
                    expires_at=now + effective_ttl,
                    created_at=now,
                )
                return

            # Evict if at capacity
            if len(self._store) >= self._max_size:
                self._evict(now)

            self._store[key] = CacheEntry(
                value=value,
                expires_at=now + effective_ttl,
                created_at=now,
            )

    def invalidate(self, key: str) -> bool:
        """Remove a single key. Returns True if the key existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all keys starting with the given prefix. Returns count removed."""
        with self._lock:
            keys_to_remove = [k for k in self._store if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._store[k]
            return len(keys_to_remove)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(
                    self._hits / max(self._hits + self._misses, 1), 3
                ),
            }

    def _evict(self, now: float) -> None:
        """Evict expired entries first, then oldest. Must be called with lock held."""
        # 1. Remove expired entries
        expired_keys = [k for k, v in self._store.items() if now > v.expires_at]
        for k in expired_keys:
            del self._store[k]
            self._evictions += 1

        # 2. If still at capacity, remove oldest entries
        while len(self._store) >= self._max_size:
            oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
            del self._store[oldest_key]
            self._evictions += 1


# ── Helper Key Functions ──────────────────────────────────


def user_profile_key(user_id: str) -> str:
    """Cache key for user profile."""
    return f"user_profile:{user_id}"


def effectiveness_key(user_id: str) -> str:
    """Cache key for effectiveness profile."""
    return f"effectiveness:{user_id}"


# ── Singleton ─────────────────────────────────────────────

_cache: Optional[TTLCache] = None


def get_cache() -> TTLCache:
    """Get or create TTLCache singleton."""
    global _cache
    if _cache is None:
        from config import get_config
        config = get_config()
        _cache = TTLCache(
            default_ttl=config.CACHE_TTL_USER_PROFILE,
            max_size=config.CACHE_MAX_SIZE,
        )
    return _cache
