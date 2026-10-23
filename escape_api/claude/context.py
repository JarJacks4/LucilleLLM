"""
"Memory since inception" — one Claude turn's context.

A Claude turn is built from exactly the same stored history as the OpenAI Lucille,
so the person never repeats themselves after switching edition:

  1. chat_sessions  (Firestore, firebase_service.py) — every message ever, both editions,
     mirrored per-user in the v1 repo so threads work without Firestore too.
  2. user_memories  (memory_service.py) — extracted facts. This edition writes new ones
     with the fast model through ClaudeMemoryExtractor, into the same collection
     and the same schema.
  3. /v1/me profile (user_service.py + the v1 state doc), latest mood check-in,
     active plan, recent journal tone.

Assembly order in the system prompt:
    identity -> profile -> memories (top 20 by importance x recency)
             -> running summary of older sessions -> last 40 turns verbatim

When the window would exceed the token budget the oldest verbatim turns are folded
INTO that running summary with the fast model. They are never silently dropped.

No `openai` import anywhere in this file: the services it reuses are called through
small accessors so a missing OpenAI key (or a missing Firestore) degrades to empty
context instead of an exception.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from escape_api import core
from escape_api.claude import client as claude
from escape_api.claude import persona
from escape_api.repo import Repo, user_path
from escape_api.settings import cfg

logger = logging.getLogger(__name__)

THREADS = "lucille_threads"
MESSAGES = "lucille_messages"

# ── service accessors (overridable in tests, never fatal) ─────────────────────

_services: Dict[str, Any] = {"memory": None, "firebase": None, "users": None}


def set_services(*, memory: Any = None, firebase: Any = None, users: Any = None) -> None:
    """Tests inject fakes. Pass nothing to clear."""
    _services["memory"] = memory
    _services["firebase"] = firebase
    _services["users"] = users


def memory_service() -> Optional[Any]:
    if _services["memory"] is not None:
        return _services["memory"]
    try:
        from memory_service import get_memory_service
        return get_memory_service()          # raises until main.py has initialised it
    except Exception as e:
        logger.debug("memory service unavailable: %s", e)
        return None


def firebase_service() -> Optional[Any]:
    if _services["firebase"] is not None:
        return _services["firebase"]
    try:
        from firebase_service import get_firebase_service
        svc = get_firebase_service()
        return svc if getattr(svc, "db", None) is not None else None
    except Exception as e:
        logger.debug("firebase service unavailable: %s", e)
        return None


def user_service() -> Optional[Any]:
    if _services["users"] is not None:
        return _services["users"]
    try:
        from user_service import get_user_service
        return get_user_service()
    except Exception as e:
        logger.debug("user service unavailable: %s", e)
        return None


# ── the assembled context ─────────────────────────────────────────────────────

@dataclass
class TurnContext:
    session_id: str
    is_new_thread: bool
    system: str
    messages: List[Dict[str, Any]]
    statuses: List[Dict[str, str]] = field(default_factory=list)
    memories: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    folded_turns: int = 0
    verbatim_turns: int = 0
    estimated_tokens: int = 0
    snapshot: Dict[str, Any] = field(default_factory=dict)
    title: Optional[str] = None

    def as_debug(self) -> Dict[str, Any]:
        return {"sessionId": self.session_id, "memories": len(self.memories),
                "verbatimTurns": self.verbatim_turns, "foldedTurns": self.folded_turns,
                "summaryChars": len(self.summary), "estimatedTokens": self.estimated_tokens,
                "statuses": [s["label"] for s in self.statuses]}


# ── 1. profile ────────────────────────────────────────────────────────────────

def profile_text(repo: Repo, uid: str, tz: str) -> str:
    """The /v1/me profile: legacy user profile (if any) + the v1 state doc."""
    lines: List[str] = []
    svc = user_service()
    if svc is not None:
        try:
            prof = svc.get_user_profile(uid)
            if prof:
                formatted = svc.format_profile_for_prompt(prof)
                if formatted:
                    lines.append(str(formatted).strip())
        except Exception as e:
            logger.debug("profile read failed for %s: %s", uid, e)
    st = core.get_state(repo, uid) or {}
    bits = []
    if st.get("displayName"):
        bits.append(f"Name they gave: {st['displayName']}")
    bits.append(f"Time zone: {tz}")
    if st.get("streakDays"):
        bits.append(f"Current streak: {st['streakDays']} days (best {st.get('bestStreak', 0)})")
    if st.get("ageBand"):
        bits.append(f"Age band: {st['ageBand']}")
    if st.get("isPremium"):
        bits.append("Premium subscriber")
    lines.append("Escape account:\n" + "\n".join(f"- {b}" for b in bits))
    return "\n\n".join(l for l in lines if l)


# ── 2. memories ───────────────────────────────────────────────────────────────

def _recency_weight(created_at: Any, now: datetime) -> float:
    """
    1.0 today, falling to a floor of 0.6 after 120 days. The floor is deliberate:
    importance has to be able to out-rank freshness, so something vital the person
    said months ago still beats today's small talk.
    """
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
    except Exception:
        return 0.8
    return max(0.6, 1.0 - min(1.0, age_days / 120.0) * 0.4)


def top_memories(uid: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Top memories by importance x recency, from the shared user_memories collection."""
    svc = memory_service()
    if svc is None:
        return []
    k = limit or cfg().claude_memory_top_k
    try:
        rows = svc.get_memories(uid, limit=max(100, k * 4)) or []
    except Exception as e:
        logger.debug("memory read failed for %s: %s", uid, e)
        return []
    now = core.now_utc()
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for m in rows:
        if not isinstance(m, dict) or not m.get("content"):
            continue
        try:
            imp = max(1, min(10, int(m.get("importance", 5))))
        except (TypeError, ValueError):
            imp = 5
        scored.append((imp * _recency_weight(m.get("created_at"), now), m))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [m for _, m in scored[:k]]


def memories_text(uid: str, memories: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
    mems = memories if memories is not None else top_memories(uid)
    svc = memory_service()
    if not mems:
        return "", []
    if svc is not None:
        try:
            return (svc.format_memories_for_prompt(mems) or ""), mems
        except Exception:
            pass
    return "\n".join(f"- {m.get('content')}" for m in mems if m.get("content")), mems


# ── 3. history: chat_sessions + per-user mirror ───────────────────────────────

def _mirror_messages(repo: Repo, uid: str, session_id: str, limit: int = 400) -> List[Dict[str, Any]]:
    rows = repo.query(user_path(uid, MESSAGES), where=[("threadId", "==", session_id)],
                      order_by="createdAt", limit=limit)
    return [d for _, d in rows]


def _firestore_messages(session_id: str) -> List[Dict[str, Any]]:
    svc = firebase_service()
    if svc is None or not session_id:
        return []
    try:
        data = svc.get_chat_session(session_id) or {}
    except Exception as e:
        logger.debug("chat_sessions read failed for %s: %s", session_id, e)
        return []
    out = []
    for m in data.get("messages") or []:
        if not isinstance(m, dict):
            continue
        kind = m.get("type") or m.get("role")
        role = "user" if kind in ("human", "user") else ("assistant" if kind in ("ai", "assistant") else None)
        if role and (m.get("content") or "").strip():
            out.append({"role": role, "text": str(m["content"]), "createdAt": m.get("timestamp")})
    return out


def load_history(repo: Repo, uid: str, session_id: Optional[str]) -> List[Dict[str, Any]]:
    """
    Every message of this thread, oldest first. Firestore `chat_sessions` is the
    shared source of truth (both editions write there); the per-user mirror is used
    when Firestore is unavailable, and to keep thread ownership checkable.
    """
    if not session_id:
        return []
    shared = _firestore_messages(session_id)
    if shared:
        return shared
    return [{"role": m.get("role"), "text": m.get("text") or "", "createdAt": m.get("createdAt")}
            for m in _mirror_messages(repo, uid, session_id)
            if m.get("role") in ("user", "assistant") and (m.get("text") or "").strip()]


def older_sessions_summary(repo: Repo, uid: str, session_id: Optional[str], limit: int = 12) -> str:
    """Running summary of earlier threads: the stored per-thread summaries, newest first."""
    parts: List[str] = []
    svc = firebase_service()
    if svc is not None:
        try:
            for s in (svc.get_user_sessions(uid, limit=limit) or []):
                sid = s.get("session_id")
                if not sid or sid == session_id:
                    continue
                summary = (s.get("summary") or "").strip()
                if summary:
                    parts.append(f"[{(s.get('updated_at') or '')[:10]}] {summary}")
        except Exception as e:
            logger.debug("session summaries read failed for %s: %s", uid, e)
    if not parts:
        rows = repo.query(user_path(uid, THREADS), order_by="updatedAt", desc=True, limit=limit)
        for tid, t in rows:
            if tid == session_id:
                continue
            summary = (t.get("summary") or "").strip()
            if summary:
                parts.append(f"[{(t.get('updatedAt') or '')[:10]}] {summary}")
    # This thread's own folded-down summary belongs at the end, closest to the verbatim turns.
    own = (repo.get(user_path(uid, THREADS, session_id)) or {}).get("summary") if session_id else None
    if own:
        parts.append(f"[this thread, earlier] {own.strip()}")
    return "\n\n".join(parts[: limit + 1])


# ── 4. the Escape snapshot (mood, plan, journal tone) ────────────────────────

def escape_snapshot(repo: Repo, uid: str, tz: str) -> Dict[str, Any]:
    """Real data only. Every field may be absent; nothing is invented."""
    snap: Dict[str, Any] = {}
    st = core.get_state(repo, uid) or {}

    rows = repo.query(user_path(uid, "mood_checkins"), order_by="createdAt", desc=True, limit=3)
    checkins = [{"word": d.get("word"), "tone": d.get("tone"), "day": d.get("localDay"),
                 "valence": d.get("valence"), "energy": d.get("energy")} for _, d in rows]
    if checkins:
        snap["checkins"] = checkins
    elif st.get("lastMood"):
        snap["checkins"] = [{"word": (st["lastMood"] or {}).get("word")}]

    entries = repo.query(user_path(uid, "journal_entries"), order_by="createdAt", desc=True, limit=5)
    saved = [d for _, d in entries if d.get("status") == "saved"]
    if saved:
        tones = [((e.get("mood") or {}).get("tone")) for e in saved if (e.get("mood") or {}).get("tone")]
        themes: List[str] = []
        for e in saved:
            for t in e.get("themes") or []:
                if t not in themes:
                    themes.append(t)
        snap["journal"] = {"entries": len(saved), "lastDay": saved[0].get("localDay"),
                           "tones": tones[:5], "themes": themes[:6],
                           "anyFlagged": any(e.get("flagged") for e in saved)}

    from escape_api import plans_claude  # local import: plans_claude imports nothing from here
    plan = plans_claude.active_plan(repo, uid)
    if plan:
        today = core.local_day(tz)
        items = [i for i in plan.get("items", []) if i.get("date") == today]
        snap["plan"] = {"id": plan.get("id"), "title": plan.get("title"), "days": plan.get("days"),
                        "minutesPerDay": plan.get("minutesPerDay"), "provider": plan.get("provider"),
                        "todayItems": [{"title": i.get("title"), "minutes": i.get("minutes"),
                                        "done": bool(i.get("done"))} for i in items],
                        "doneTotal": sum(1 for i in plan.get("items", []) if i.get("done")),
                        "itemsTotal": len(plan.get("items", []))}

    snap["streakDays"] = int(st.get("streakDays") or 0)
    snap["coins"] = int(st.get("coins") or 0)
    snap["today"] = core.local_day(tz)
    return snap


def snapshot_text(snap: Dict[str, Any]) -> str:
    """Render the snapshot for the prompt. Says 'none yet' rather than leaving a gap."""
    lines: List[str] = [f"Today (their time zone): {snap.get('today')}"]
    ck = snap.get("checkins") or []
    if ck:
        lines.append("Recent mood check-ins (newest first): "
                     + "; ".join(f"{c.get('word') or '?'}{' on ' + c['day'] if c.get('day') else ''}" for c in ck))
    else:
        lines.append("Mood check-ins: none recorded yet.")
    j = snap.get("journal")
    if j:
        lines.append(f"Journal: {j['entries']} recent entries, last on {j.get('lastDay')}; "
                     f"tone {', '.join(j.get('tones') or []) or 'unclear'}; "
                     f"themes {', '.join(j.get('themes') or []) or 'none tagged'}.")
    else:
        lines.append("Journal: nothing written yet.")
    p = snap.get("plan")
    if p:
        todo = [i["title"] for i in p.get("todayItems", []) if not i["done"]]
        lines.append(f"Active plan '{p.get('title')}' — {p.get('days')} days, {p.get('minutesPerDay')} min/day, "
                     f"{p.get('doneTotal')} of {p.get('itemsTotal')} steps done. "
                     + (f"Left today: {', '.join(todo)}." if todo else "Nothing left for today."))
    else:
        lines.append("Self-care plan: none active.")
    lines.append(f"Streak: {snap.get('streakDays', 0)} days. Coins: {snap.get('coins', 0)}.")
    return "\n".join(lines)


# ── 5. fold-down with the fast model ─────────────────────────────────────────

def _as_transcript(turns: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{'Person' if t.get('role') == 'user' else 'Lucille'}: {t.get('text') or ''}"
                     for t in turns)


def deterministic_fold(turns: List[Dict[str, Any]], previous: str = "") -> str:
    """
    Fold-down without a model: keep the person's own lines, trimmed. Used when the
    fast model is unavailable, so old turns are still never dropped.
    """
    said = [f"- {(t.get('text') or '').strip()[:220]}" for t in turns if t.get("role") == "user"]
    block = "Earlier in this conversation the person said:\n" + "\n".join(said[-40:]) if said else ""
    return "\n\n".join(p for p in (previous.strip(), block) if p).strip()


async def fold_history(turns: List[Dict[str, Any]], previous: str = "") -> str:
    """Summarise `turns` into the running summary with the fast model."""
    if not turns:
        return previous
    prompt = (("Existing notes:\n" + previous + "\n\n") if previous else "") + \
        "<conversation>\n" + _as_transcript(turns)[:120_000] + "\n</conversation>\n\n" + \
        "Update the notes so nothing the person told you is lost."
    text = await claude.one_shot(persona.FOLD_SYSTEM, prompt,
                                 model=cfg().claude_fast_model, max_tokens=700, fallback="")
    return text.strip() if text.strip() else deterministic_fold(turns, previous)


# ── 6. build a turn ──────────────────────────────────────────────────────────

def _to_messages(turns: List[Dict[str, Any]], user_message: Optional[str]) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    for t in turns:
        role = t.get("role")
        text = (t.get("text") or "").strip()
        if role not in ("user", "assistant") or not text:
            continue
        if msgs and msgs[-1]["role"] == role:          # the API wants alternating roles
            msgs[-1]["content"] += "\n\n" + text
        else:
            msgs.append({"role": role, "content": text})
    if user_message:
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] += "\n\n" + user_message
        else:
            msgs.append({"role": "user", "content": user_message})
    if msgs and msgs[0]["role"] != "user":
        msgs.insert(0, {"role": "user", "content": "(earlier conversation)"})
    return msgs


async def build_turn_context(repo: Repo, uid: str, tz: str, *, message: Optional[str],
                             session_id: Optional[str] = None,
                             client_context: Optional[Dict[str, Any]] = None,
                             safety_override: str = "",
                             with_tools: bool = True) -> TurnContext:
    """Assemble one Claude turn from everything this person has ever stored."""
    is_new = not session_id
    sid = session_id or core.new_id("luc")
    statuses: List[Dict[str, str]] = []

    prof = profile_text(repo, uid, tz)
    mem_text, mems = memories_text(uid)
    history = load_history(repo, uid, sid)
    summary = older_sessions_summary(repo, uid, sid)
    snap = escape_snapshot(repo, uid, tz)

    # honest statuses: only what actually happened
    if mems:
        statuses.append({"kind": "reading", "label": f"Read {len(mems)} things you've told me"})
    ck = snap.get("checkins") or []
    if ck:
        statuses.append({"kind": "reading",
                         "label": f"Read your last {len(ck)} check-in{'s' if len(ck) != 1 else ''}"})
    if history:
        statuses.append({"kind": "reading", "label": f"Picked up where we left off ({len(history)} messages)"})

    verbatim_limit = max(2, cfg().claude_verbatim_turns)
    folded_count = 0
    older, recent = [], history
    if len(history) > verbatim_limit:
        older, recent = history[:-verbatim_limit], history[-verbatim_limit:]

    def assemble(summary_text: str, turns: List[Dict[str, Any]]):
        system = persona.system_prompt(
            profile_text=prof, memories_text=mem_text, summary_text=summary_text,
            escape_state_text=snapshot_text(snap),
            client_context_text=persona.client_context_text(client_context),
            safety_override=safety_override, with_tools=with_tools,
        )
        return system, _to_messages(turns, message)

    if older:
        summary = await fold_history(older, summary)
        folded_count = len(older)

    system, messages = assemble(summary, recent)
    est = claude.estimate_messages_tokens(system, messages)

    # Budget fold-down: keep halving the verbatim window until it fits. Folded turns
    # go into the running summary, never into the bin.
    budget = max(20_000, cfg().claude_context_budget_tokens)
    while est > budget and len(recent) > 2:
        cut = max(1, len(recent) // 2)
        summary = await fold_history(recent[:cut], summary)
        folded_count += cut
        recent = recent[cut:]
        system, messages = assemble(summary, recent)
        est = claude.estimate_messages_tokens(system, messages)

    if folded_count:
        statuses.append({"kind": "reading",
                         "label": f"Folded {folded_count} older messages into my notes"})
        if session_id:
            repo.set(user_path(uid, THREADS, sid), {"summary": summary, "foldedTurns": folded_count}, merge=True)

    return TurnContext(session_id=sid, is_new_thread=is_new, system=system, messages=messages,
                       statuses=statuses, memories=mems, summary=summary, folded_turns=folded_count,
                       verbatim_turns=len(recent), estimated_tokens=est, snapshot=snap)


# ── 7. writing the turn back (both editions read it) ─────────────────────────

def thread_doc(repo: Repo, uid: str, session_id: str) -> Optional[Dict[str, Any]]:
    return repo.get(user_path(uid, THREADS, session_id))


def list_threads(repo: Repo, uid: str, limit: int = 50) -> List[Dict[str, Any]]:
    rows = repo.query(user_path(uid, THREADS), order_by="updatedAt", desc=True, limit=limit)
    return [{"sessionId": tid, "title": d.get("title"), "createdAt": d.get("createdAt"),
             "updatedAt": d.get("updatedAt"), "messages": int(d.get("messageCount") or 0),
             "edition": d.get("edition", "claude")}
            for tid, d in rows if not d.get("deletedAt")]


def thread_messages(repo: Repo, uid: str, session_id: str, limit: int = 400) -> List[Dict[str, Any]]:
    msgs = load_history(repo, uid, session_id)
    return [{"role": m.get("role"), "text": m.get("text"), "createdAt": m.get("createdAt")} for m in msgs][-limit:]


def _write_shared_session(uid: str, session_id: str, history: List[Dict[str, Any]], summary: str) -> bool:
    """Mirror the thread into Firestore `chat_sessions` so the OpenAI edition sees it too."""
    svc = firebase_service()
    if svc is None:
        return False
    try:
        from langchain_core.messages import AIMessage, HumanMessage
        msgs = [HumanMessage(content=m["text"]) if m["role"] == "user" else AIMessage(content=m["text"])
                for m in history if (m.get("text") or "").strip()]
        existing = svc.get_chat_session(session_id)
        if existing:
            svc.update_chat_session(session_id, msgs, summary or (existing.get("summary") or ""))
        else:
            svc.store_chat_session(session_id, msgs, summary or "")
        svc.link_session_to_user(session_id, uid)
        return True
    except Exception as e:
        logger.warning("chat_sessions write failed for %s: %s", session_id, e)
        return False


def record_turn(repo: Repo, uid: str, tz: str, session_id: str, user_text: str, assistant_text: str,
                *, title: Optional[str] = None, summary: str = "") -> Dict[str, Any]:
    """
    Persist one turn. Writes the per-user mirror (always) and `chat_sessions`
    (when Firestore is up) so both editions read the same history.
    """
    now = core.iso(core.now_utc())
    before = thread_doc(repo, uid, session_id) or {}
    for role, text in (("user", user_text), ("assistant", assistant_text)):
        if not (text or "").strip():
            continue
        mid = core.new_id("lm")
        repo.set(user_path(uid, MESSAGES, mid),
                 {"id": mid, "threadId": session_id, "role": role, "text": text,
                  "createdAt": now, "localDay": core.local_day(tz)})
    count = int(before.get("messageCount") or 0) + sum(1 for t in (user_text, assistant_text) if (t or "").strip())
    doc = {"id": session_id, "title": title or before.get("title") or _fallback_title(user_text),
           "createdAt": before.get("createdAt") or now, "updatedAt": now,
           "messageCount": count, "turns": int(before.get("turns") or 0) + 1,
           "edition": "claude", "deletedAt": None}
    if summary:
        doc["summary"] = summary
    repo.set(user_path(uid, THREADS, session_id), doc, merge=True)
    shared = _write_shared_session(uid, session_id, load_history(repo, uid, session_id) or
                                   [{"role": "user", "text": user_text}, {"role": "assistant", "text": assistant_text}],
                                   summary or before.get("summary") or "")
    return {"thread": doc, "shared": shared}


def delete_thread(repo: Repo, uid: str, session_id: str) -> int:
    n = 0
    for mid, _ in repo.query(user_path(uid, MESSAGES), where=[("threadId", "==", session_id)]):
        repo.delete(user_path(uid, MESSAGES, mid))
        n += 1
    repo.delete(user_path(uid, THREADS, session_id))
    svc = firebase_service()
    if svc is not None:
        try:
            svc.delete_chat_session(session_id)
        except Exception as e:
            logger.warning("chat_sessions delete failed for %s: %s", session_id, e)
    return n


def _fallback_title(user_text: str) -> str:
    words = (user_text or "").strip().split()
    return " ".join(words[:6]) if words else "New conversation"


async def make_title(user_text: str) -> str:
    """Title a new thread with the fast model; the person's own first words otherwise."""
    fallback = _fallback_title(user_text)
    if not (user_text or "").strip():
        return fallback
    text = await claude.one_shot(persona.TITLE_SYSTEM, f"<conversation>\n{user_text[:2000]}\n</conversation>",
                                 model=cfg().claude_fast_model, max_tokens=24, fallback=fallback)
    cleaned = " ".join(text.replace("\n", " ").strip().strip('".').split())[:60]
    return cleaned or fallback


# ── 8. memory extraction with the fast model ─────────────────────────────────

VALID_MEMORY_TYPES = ("episodic", "semantic", "fact")


class ClaudeMemoryExtractor:
    """
    Writes new memories into the SAME collection and schema as the OpenAI edition
    (memory_service.store_memory), using the fast Claude model instead of GPT.
    Silent no-op when Claude or the memory service is unavailable.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or cfg().claude_fast_model

    async def extract(self, uid: str, user_message: str, assistant_response: str) -> List[str]:
        if not cfg().claude_memory_write or not claude.available():
            return []
        svc = memory_service()
        if svc is None:
            return []
        convo = (f"<conversation>\nPerson: {(user_message or '')[:4000]}\n"
                 f"Lucille: {(assistant_response or '')[:4000]}\n</conversation>")
        raw = await claude.one_shot(persona.MEMORY_SYSTEM, convo, model=self.model, max_tokens=500, fallback="")
        stored: List[str] = []
        for item in self._parse(raw):
            try:
                mid = svc.store_memory(uid, item)
            except Exception as e:
                logger.debug("store_memory failed: %s", e)
                mid = None
            if mid:
                stored.append(mid)
        if stored:
            logger.info("Claude edition stored %d memories for %s", len(stored), uid)
        return stored

    @staticmethod
    def _parse(raw: str) -> List[Dict[str, Any]]:
        if not raw:
            return []
        try:
            start = min([i for i in (raw.find("{"), raw.find("[")) if i >= 0], default=-1)
            if start < 0:
                return []
            end = max(raw.rfind("}"), raw.rfind("]"))
            data = json.loads(raw[start:end + 1])
        except Exception:
            return []
        items = data.get("memories", data.get("items", [])) if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        out: List[Dict[str, Any]] = []
        for m in items[:8]:
            if not isinstance(m, dict) or not (m.get("content") or "").strip():
                continue
            mem_type = m.get("memory_type") if m.get("memory_type") in VALID_MEMORY_TYPES else "semantic"
            try:
                importance = max(1, min(10, int(m.get("importance", 5))))
            except (TypeError, ValueError):
                importance = 5
            tags = [str(t)[:24] for t in (m.get("tags") or []) if isinstance(t, (str, int))][:3]
            out.append({"content": str(m["content"])[:600], "memory_type": mem_type,
                        "importance": importance, "tags": tags,
                        "source": "claude_extracted", "edition": "claude"})
        return out


# ── 9. the periodic AI disclosure ────────────────────────────────────────────

def disclosure_due(repo: Repo, uid: str) -> bool:
    """
    Contract cadence: adults every 3 h or 12 turns, teens every 1 h or 5 turns.
    Also always on the very first turn we ever answer for this person.
    """
    s = cfg()
    st = core.get_state(repo, uid) or {}
    age_band = str(st.get("ageBand") or "")
    teen = bool(st.get("isMinor")) or age_band in ("13-17", "under13") or age_band == ""
    hours = s.disclosure_hours_teen if teen else s.disclosure_hours_adult
    turns = s.disclosure_turns_teen if teen else s.disclosure_turns_adult
    d = repo.get(user_path(uid, "state", "lucille_claude")) or {}
    last = d.get("lastDisclosureAt")
    if not last:
        return True
    since_turns = int(d.get("turnsSinceDisclosure") or 0)
    if since_turns >= turns:
        return True
    try:
        when = core.parse_iso(last)
    except Exception:
        return True
    return core.now_utc() - when >= timedelta(hours=hours)


def note_turn(repo: Repo, uid: str, *, disclosed: bool) -> None:
    path = user_path(uid, "state", "lucille_claude")
    d = repo.get(path) or {}
    upd = {"turnsSinceDisclosure": 0 if disclosed else int(d.get("turnsSinceDisclosure") or 0) + 1,
           "lastTurnAt": core.iso(core.now_utc()),
           "turnsTotal": int(d.get("turnsTotal") or 0) + 1}
    if disclosed:
        upd["lastDisclosureAt"] = core.iso(core.now_utc())
    repo.set(path, upd, merge=True)
