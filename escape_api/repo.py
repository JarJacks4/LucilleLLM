"""
Storage layer for the Escape v1 API.

All v1 data for a user lives under ONE root document:

    escape_users/{uid}
        state               (doc)   streak, coins, energy level, prefs, timezone
        consents            (doc)   GDPR consent flags + history
        journal_entries/    (coll)
        journal_letters/    (coll)
        mood_checkins/      (coll)
        mood_days/          (coll)  daily rollups  -> cheap Week/Month stats
        mood_months/        (coll)  monthly rollups -> cheap Year stats
        listening_sessions/ (coll)
        library/            (coll)  saved soundscapes
        compositions/       (coll)  Lucille Compose renders
        plans/              (coll)  self-care plans
        scores/             (coll)  daily Self-Care Score snapshots
        weekly/             (coll)  Lucille weekly reflections (1 LLM call / user / week)
        usage/              (coll)  per-day quota counters

One root means GDPR export/delete is a single tree walk, and the
FlutterFlow `Users` schema is never touched.

Two implementations share one small interface:
  * FirestoreRepo  - production (firebase_admin)
  * MemoryRepo     - tests and local dev without credentials
"""

from __future__ import annotations

import copy
import logging
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT = "escape_users"

Where = Tuple[str, str, Any]  # (field, op, value)  op in ==, <, <=, >, >=, in, array_contains


def user_path(uid: str, *parts: str) -> str:
    return "/".join([ROOT, uid, *parts])


class Repo:
    """Interface. Paths are slash-joined ('escape_users/u1/journal_entries/abc')."""

    def get(self, path: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def set(self, path: str, data: Dict[str, Any], merge: bool = False) -> None:
        raise NotImplementedError

    def delete(self, path: str) -> None:
        raise NotImplementedError

    def increment(self, path: str, deltas: Dict[str, float], extra: Optional[Dict[str, Any]] = None) -> None:
        """Atomic numeric increments. Dotted keys address nested maps ('words.Calm')."""
        raise NotImplementedError

    def query(
        self,
        coll_path: str,
        where: Iterable[Where] = (),
        order_by: Optional[str] = None,
        desc: bool = False,
        limit: Optional[int] = None,
        start_after: Optional[Any] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        raise NotImplementedError

    def list_subcollections(self, doc_path: str) -> List[str]:
        raise NotImplementedError

    def collection_group(self, name: str, where: Iterable[Where] = (), limit: Optional[int] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """Query every subcollection called `name` (e.g. all users' journal_letters). Returns (full_path, data)."""
        raise NotImplementedError

    # ── helpers built on the primitives ──
    def delete_tree(self, doc_path: str, batch: int = 200) -> int:
        """Delete a document and everything beneath it. Returns docs deleted."""
        n = 0
        for sub in self.list_subcollections(doc_path):
            coll = f"{doc_path}/{sub}"
            for doc_id, _ in self.query(coll):
                n += self.delete_tree(f"{coll}/{doc_id}", batch)
        if self.get(doc_path) is not None:
            self.delete(doc_path)
            n += 1
        return n

    def dump_tree(self, doc_path: str) -> Dict[str, Any]:
        """Export a document and its subcollections as nested JSON."""
        out: Dict[str, Any] = {"_doc": self.get(doc_path)}
        for sub in self.list_subcollections(doc_path):
            coll = f"{doc_path}/{sub}"
            out[sub] = {doc_id: self.dump_tree(f"{coll}/{doc_id}") for doc_id, _ in self.query(coll)}
        return out


# ─────────────────────────── Memory ───────────────────────────

def _cmp(a: Any, op: str, b: Any) -> bool:
    try:
        if op == "==":
            return a == b
        if op == "!=":
            return a != b
        if op == "<":
            return a is not None and a < b
        if op == "<=":
            return a is not None and a <= b
        if op == ">":
            return a is not None and a > b
        if op == ">=":
            return a is not None and a >= b
        if op == "in":
            return a in b
        if op == "array_contains":
            return isinstance(a, list) and b in a
    except TypeError:
        return False
    raise ValueError(f"unsupported op {op}")


def _get_dotted(d: Dict[str, Any], key: str) -> Any:
    cur: Any = d
    for k in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _set_dotted(d: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for k in parts[:-1]:
        cur = cur.setdefault(k, {})
    cur[parts[-1]] = value


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


class MemoryRepo(Repo):
    def __init__(self) -> None:
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def get(self, path):
        with self._lock:
            d = self._docs.get(path)
            return copy.deepcopy(d) if d is not None else None

    def set(self, path, data, merge=False):
        with self._lock:
            if merge and path in self._docs:
                _deep_merge(self._docs[path], data)
            else:
                self._docs[path] = copy.deepcopy(data)

    def delete(self, path):
        with self._lock:
            self._docs.pop(path, None)

    def increment(self, path, deltas, extra=None):
        with self._lock:
            doc = self._docs.setdefault(path, {})
            for k, v in deltas.items():
                _set_dotted(doc, k, (_get_dotted(doc, k) or 0) + v)
            if extra:
                _deep_merge(doc, extra)

    def query(self, coll_path, where=(), order_by=None, desc=False, limit=None, start_after=None):
        with self._lock:
            depth = coll_path.count("/") + 1
            rows = [
                (p.rsplit("/", 1)[1], copy.deepcopy(d))
                for p, d in self._docs.items()
                if p.startswith(coll_path + "/") and p.count("/") == depth
            ]
        for f, op, v in where:
            rows = [r for r in rows if _cmp(_get_dotted(r[1], f), op, v)]
        if order_by:
            rows.sort(key=lambda r: (_get_dotted(r[1], order_by) is None, _get_dotted(r[1], order_by)), reverse=desc)
            if start_after is not None:
                rows = [r for r in rows if _cmp(_get_dotted(r[1], order_by), "<" if desc else ">", start_after)]
        else:
            rows.sort(key=lambda r: r[0])
        return rows[:limit] if limit else rows

    def collection_group(self, name, where=(), limit=None):
        with self._lock:
            rows = [(p, copy.deepcopy(d)) for p, d in self._docs.items() if p.split("/")[-2:-1] == [name]]
        for f, op, v in where:
            rows = [r for r in rows if _cmp(_get_dotted(r[1], f), op, v)]
        return rows[:limit] if limit else rows

    def list_subcollections(self, doc_path):
        with self._lock:
            prefix = doc_path + "/"
            subs = {p[len(prefix):].split("/", 1)[0] for p in self._docs if p.startswith(prefix)}
        return sorted(subs)


# ─────────────────────────── Firestore ───────────────────────────

class FirestoreRepo(Repo):
    def __init__(self, db) -> None:
        from firebase_admin import firestore  # noqa: F401  (import check)
        self.db = db

    def _doc(self, path):
        return self.db.document(path)

    def get(self, path):
        snap = self._doc(path).get()
        return snap.to_dict() if snap.exists else None

    def set(self, path, data, merge=False):
        self._doc(path).set(data, merge=merge)

    def delete(self, path):
        self._doc(path).delete()

    def increment(self, path, deltas, extra=None):
        from google.cloud.firestore_v1 import Increment
        payload: Dict[str, Any] = {}
        for k, v in deltas.items():
            _set_dotted(payload, k, Increment(v))
        if extra:
            _deep_merge(payload, extra)
        self._doc(path).set(payload, merge=True)

    def query(self, coll_path, where=(), order_by=None, desc=False, limit=None, start_after=None):
        from google.cloud.firestore_v1 import FieldFilter, Query
        q = self.db.collection(coll_path)
        for f, op, v in where:
            q = q.where(filter=FieldFilter(f, op, v))
        if order_by:
            q = q.order_by(order_by, direction=Query.DESCENDING if desc else Query.ASCENDING)
            if start_after is not None:
                q = q.start_after({order_by: start_after})
        if limit:
            q = q.limit(limit)
        return [(s.id, s.to_dict()) for s in q.stream()]

    def list_subcollections(self, doc_path):
        return [c.id for c in self._doc(doc_path).collections()]

    def collection_group(self, name, where=(), limit=None):
        from google.cloud.firestore_v1 import FieldFilter
        q = self.db.collection_group(name)
        for f, op, v in where:
            q = q.where(filter=FieldFilter(f, op, v))
        if limit:
            q = q.limit(limit)
        return [(s.reference.path, s.to_dict()) for s in q.stream()]

    def delete_tree(self, doc_path, batch=200):
        # Firestore has a server-side recursive delete; much cheaper than walking.
        try:
            return self.db.recursive_delete(self._doc(doc_path), chunk_size=batch) or 0
        except Exception as e:  # pragma: no cover - fallback path
            logger.warning(f"recursive_delete failed ({e}); walking tree")
            return super().delete_tree(doc_path, batch)


# ─────────────────────────── selection ───────────────────────────

_repo: Optional[Repo] = None


def get_repo() -> Repo:
    """Production: Firestore via the existing FirebaseService. Falls back to memory (dev)."""
    global _repo
    if _repo is None:
        try:
            from firebase_service import get_firebase_service
            db = get_firebase_service().db
        except Exception as e:  # pragma: no cover
            logger.warning(f"Firebase unavailable for v1 API: {e}")
            db = None
        if db is not None and not hasattr(db, "_mock_children"):
            _repo = FirestoreRepo(db)
        else:
            logger.warning("v1 API using in-memory storage (no Firestore). Data will not persist.")
            _repo = MemoryRepo()
    return _repo


def set_repo(repo: Optional[Repo]) -> None:
    """Tests inject a MemoryRepo here."""
    global _repo
    _repo = repo
