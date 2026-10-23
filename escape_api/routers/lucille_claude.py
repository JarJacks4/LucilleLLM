"""
/v2/lucille — the Claude edition's conversation.

    POST   /v2/lucille/turn           one turn, streamed as Server-Sent Events
    POST   /v2/lucille/turn/audio     the same turn from a recording (fallback path)
    POST   /v2/lucille/approve        Allow / Not now for anything that writes
    GET    /v2/lucille/threads        the person's threads
    GET    /v2/lucille/threads/{id}   one thread's messages
    DELETE /v2/lucille/threads/{id}   delete a thread (and its shared chat_session)

Event types are exactly the contract's: session, status, delta, artifact, action,
plan, approval, safety, disclosure, done, error.

Rules kept here, not in the model:
  * a Claude turn needs `ai_reflection` consent; plan tools additionally need
    `personalization`
  * distress is screened server-side (again) before anything leaves this process,
    and crisis wording comes from safety_service — never from Claude
  * anything that writes outside the conversation emits `approval` and waits for
    POST /v2/lucille/approve; nothing is written on a timeout
  * the AI disclosure cadence is adults 3 h / 12 turns, teens 1 h / 5 turns
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from escape_api import core, tts
from escape_api.claude import client as claude
from escape_api.claude import context as ctx_mod
from escape_api.claude import tools as tools_mod
from escape_api.deps import Caller, caller, has_consent, require_consent
from escape_api.settings import cfg

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/lucille", tags=["v2 · Lucille (Claude)"])

SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
               "Connection": "keep-alive"}


class TurnIn(BaseModel):
    message: Optional[str] = Field(None, max_length=8000)
    sessionId: Optional[str] = Field(None, max_length=64)
    mode: Literal["text", "voice"] = "text"
    speak: bool = False
    clientContext: Optional[Dict[str, Any]] = None


class ApproveIn(BaseModel):
    id: str = Field(..., max_length=64)
    allow: bool


# ── SSE plumbing ─────────────────────────────────────────────────────────────

def sse(event: Dict[str, Any]) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False, default=str) + "\n\n"


def _status(kind: str, label: str) -> Dict[str, Any]:
    return {"type": "status", "kind": kind, "label": label}


def _stream(gen: AsyncIterator[Dict[str, Any]]) -> StreamingResponse:
    async def body():
        async for ev in gen:
            yield sse(ev)
    return StreamingResponse(body(), media_type="text/event-stream", headers=SSE_HEADERS)


# ── guards shared by both turn endpoints ─────────────────────────────────────

def _guard(c: Caller, message: Optional[str]) -> None:
    require_consent(c, "ai_reflection")
    if not (message or "").strip():
        raise HTTPException(422, {"code": "empty_message", "message": "Say something first."})
    if not claude.available():
        raise HTTPException(503, claude.ClaudeUnavailable().as_detail())
    if not core.check_quota(c.repo, c.uid, c.tz, "claude_turn", cfg().quota_claude_turn_per_day):
        raise HTTPException(429, {"code": "quota_exceeded",
                                  "message": "You've reached today's limit for Lucille. She'll be back tomorrow."})


def _tool_ctx(c: Caller, session_id: str) -> tools_mod.ToolCtx:
    return tools_mod.ToolCtx(
        repo=c.repo, uid=c.uid, tz=c.tz, session_id=session_id,
        has_personalization=has_consent(c, "personalization"),
        quota_ok=lambda kind, limit: core.check_quota(c.repo, c.uid, c.tz, kind, limit),
    )


# ── the turn ─────────────────────────────────────────────────────────────────

async def _crisis_turn(c: Caller, message: str, session_id: Optional[str], safety: Dict[str, Any],
                       speak: bool) -> AsyncIterator[Dict[str, Any]]:
    """Distress never reaches the model. Wording and resources come from safety_service."""
    sid = session_id or core.new_id("luc")
    title = (ctx_mod.thread_doc(c.repo, c.uid, sid) or {}).get("title") or "Checking in"
    yield {"type": "session", "sessionId": sid, "title": title}
    reply = _crisis_text()
    yield {"type": "safety", "level": "urgent", "resources": safety.get("resources") or []}
    yield {"type": "delta", "text": reply}
    _log_safety(c, sid, safety)
    ctx_mod.record_turn(c.repo, c.uid, c.tz, sid, message, reply, title=title)
    ctx_mod.note_turn(c.repo, c.uid, disclosed=True)
    yield {"type": "disclosure", "text": core.AI_DISCLOSURE}
    done: Dict[str, Any] = {"type": "done", "messageId": core.new_id("msg"),
                            "usage": {"model": None, "inputTokens": 0, "outputTokens": 0},
                            "safety": "urgent"}
    if speak:
        audio = await _speak(reply)
        if audio:
            done |= audio
    yield done


def _crisis_text() -> str:
    try:
        from safety_service import get_safety_service
        text = (get_safety_service().get_crisis_intercept_response() or "").strip()
        if text:
            return text
    except Exception as e:
        logger.warning("crisis intercept text unavailable: %s", e)
    # Same wording the v1 voice turn uses; still not model-generated.
    from escape_api.routers.voice import DISTRESS_REPLY
    return DISTRESS_REPLY


def _log_safety(c: Caller, session_id: str, safety: Dict[str, Any]) -> None:
    try:
        from models import RiskLevel, SafetyEventType
        from safety_service import get_safety_service
        get_safety_service().log_safety_event(
            user_id=c.uid, session_id=f"lucille_claude:{session_id}",
            event_type=SafetyEventType.CRISIS_DETECTED,
            risk_level=RiskLevel(safety.get("riskLevel") or "high"),
            message_snippet="[lucille claude turn]", flags=safety.get("flags", []),
            action_taken="crisis_resources_shown")
    except Exception:
        pass


async def _speak(text: str) -> Optional[Dict[str, Any]]:
    try:
        res = await tts.synthesize(text[: tts.MAX_CHARS])
    except Exception as e:
        logger.warning("lucille claude tts failed: %s", e)
        return None
    if not res:
        return None
    return {"audio": tts.b64(res[0]), "audioMime": "audio/mpeg", "voiceProvider": res[1]}


async def _turn_events(c: Caller, body: TurnIn, message: str) -> AsyncIterator[Dict[str, Any]]:
    safety = core.safety_check(message)
    if safety.get("crisis"):
        async for ev in _crisis_turn(c, message, body.sessionId, safety, body.speak):
            yield ev
        return

    sid = body.sessionId or core.new_id("luc")
    existing = ctx_mod.thread_doc(c.repo, c.uid, sid) or {}
    title = existing.get("title") or await ctx_mod.make_title(message)
    yield {"type": "session", "sessionId": sid, "title": title}
    yield _status("thinking", "Thinking…")

    allow_personalization = has_consent(c, "personalization")
    # Always pass the id: a thread the OpenAI edition started has history in
    # chat_sessions even though this edition has never seen it.
    turn = await ctx_mod.build_turn_context(
        c.repo, c.uid, c.tz, message=message, session_id=sid if body.sessionId else None,
        client_context=body.clientContext,
    )
    for s in turn.statuses:
        yield _status(s["kind"], s["label"])

    specs = tools_mod.specs(allow_writes=True, allow_personalization=allow_personalization)
    messages: List[Dict[str, Any]] = list(turn.messages)
    tctx = _tool_ctx(c, sid)
    reply_parts: List[str] = []
    usage: Dict[str, Any] = {"model": cfg().claude_model, "inputTokens": 0, "outputTokens": 0}
    awaiting_approval = False
    client = claude.get_client()

    for round_no in range(max(1, cfg().claude_tool_rounds)):
        tool_calls: List[Dict[str, Any]] = []
        text_this_round: List[str] = []
        stop_reason = None
        try:
            async for ev in client.stream(turn.system, messages, tools=specs, model=cfg().claude_model):
                t = ev.get("type")
                if t == "text":
                    text_this_round.append(ev["text"])
                    yield {"type": "delta", "text": ev["text"]}
                elif t == "tool_use":
                    tool_calls.append(ev)
                elif t == "usage":
                    usage["inputTokens"] = (usage["inputTokens"] or 0) + (ev.get("inputTokens") or 0)
                    usage["outputTokens"] = (usage["outputTokens"] or 0) + (ev.get("outputTokens") or 0)
                elif t == "stop":
                    stop_reason = ev.get("reason")
                elif t == "error":
                    yield ev
        except claude.ClaudeError as e:
            yield e.as_event()
            break
        except Exception as e:  # noqa: BLE001
            logger.warning("lucille claude stream failed: %s: %s", type(e).__name__, e)
            yield {"type": "error", "code": "claude_error", "message": "Lucille's answer stopped early."}
            break

        reply_parts.extend(text_this_round)
        if stop_reason == "max_tokens":
            # Say it was cut off rather than letting a half sentence look finished.
            yield {"type": "error", "code": "answer_truncated",
                   "message": "Lucille ran out of room mid-answer. Ask her to carry on."}
        if not tool_calls:
            break

        assistant_blocks: List[Dict[str, Any]] = []
        if any(t.strip() for t in text_this_round):
            assistant_blocks.append({"type": "text", "text": "".join(text_this_round)})
        for call in tool_calls:
            assistant_blocks.append({"type": "tool_use", "id": call["id"], "name": call["name"],
                                     "input": call.get("input") or {}})
        messages.append({"role": "assistant", "content": assistant_blocks})

        results: List[Dict[str, Any]] = []
        for call in tool_calls:
            tool = tools_mod.get(call["name"])
            if tool is None:
                results.append({"type": "tool_result", "tool_use_id": call["id"],
                                "content": json.dumps({"error": "unknown_tool"})})
                continue
            if tool.needs_personalization and not allow_personalization:
                results.append({"type": "tool_result", "tool_use_id": call["id"],
                                "content": json.dumps({"error": "consent_required",
                                                       "purpose": "personalization",
                                                       "note": "Tell them this needs the personalisation "
                                                               "permission in Privacy, and why."})})
                continue
            if tool.writes:
                ev = tools_mod.approval_event(c.repo, c.uid, sid, tool, call.get("input") or {})
                yield ev
                awaiting_approval = True
                results.append({"type": "tool_result", "tool_use_id": call["id"],
                                "content": json.dumps({"status": "awaiting_approval",
                                                       "approvalId": ev["id"],
                                                       "note": "Nothing has been written yet. Tell them in one "
                                                               "short sentence what you want to do and why, "
                                                               "then stop and wait."})})
                continue
            if tool.pending_label:
                yield _status(tool.kind, tool.pending_label + "…")
            outcome = await tools_mod.run(tctx, call["name"], call.get("input") or {})
            if outcome.status:
                yield _status(outcome.status["kind"], outcome.status["label"])
            for extra in outcome.events:
                yield extra
            results.append({"type": "tool_result", "tool_use_id": call["id"],
                            "content": json.dumps(outcome.result, default=str)[:20000]})
        messages.append({"role": "user", "content": results})
        if awaiting_approval:
            # One more pass so she can say what she'd like to do; then stop.
            specs = tools_mod.specs(allow_writes=False, allow_personalization=allow_personalization)

    reply = "".join(reply_parts).strip()
    if not reply and not awaiting_approval:
        reply = "I'm here. Tell me a little more about that."
        yield {"type": "delta", "text": reply}

    disclosed = ctx_mod.disclosure_due(c.repo, c.uid)
    if disclosed:
        yield {"type": "disclosure", "text": core.AI_DISCLOSURE}
    ctx_mod.note_turn(c.repo, c.uid, disclosed=disclosed)

    ctx_mod.record_turn(c.repo, c.uid, c.tz, sid, message, reply, title=title, summary=turn.summary)
    try:
        await ctx_mod.ClaudeMemoryExtractor().extract(c.uid, message, reply)
    except Exception as e:  # noqa: BLE001
        logger.warning("claude memory extraction failed: %s", e)

    done: Dict[str, Any] = {"type": "done", "messageId": core.new_id("msg"), "usage": usage,
                            "awaitingApproval": awaiting_approval, "context": turn.as_debug()}
    if body.speak and reply:
        audio = await _speak(reply)
        if audio:
            done |= audio
    yield done


@router.post("/turn", summary="One Lucille turn, streamed (text/event-stream)")
async def turn(body: TurnIn, c: Caller = Depends(caller)):
    _guard(c, body.message)
    return _stream(_turn_events(c, body, (body.message or "").strip()))


@router.post("/turn/audio", summary="One Lucille turn from a recording (the phone normally transcribes)")
async def turn_audio(file: UploadFile = File(...), sessionId: str = Form(""), mode: str = Form("voice"),
                     speak: bool = Form(True), message: str = Form(""),
                     clientContext: str = Form(""), c: Caller = Depends(caller)):
    require_consent(c, "ai_reflection")
    require_consent(c, "voice")
    from escape_api.routers.voice import CONVERSE_DAILY_LIMIT, _read_audio, _whisper
    name, data = await _read_audio(file)
    if not core.check_quota(c.repo, c.uid, c.tz, "converse", CONVERSE_DAILY_LIMIT):
        raise HTTPException(429, "Daily voice limit reached. Try again tomorrow.")
    text = (message or "").strip() or await _whisper(name, data)
    try:
        client_ctx = json.loads(clientContext) if clientContext.strip() else None
    except json.JSONDecodeError:
        client_ctx = None
    body = TurnIn(message=text, sessionId=sessionId or None, mode="voice", speak=bool(speak),
                  clientContext=client_ctx if isinstance(client_ctx, dict) else None)
    _guard(c, body.message)
    return _stream(_turn_events(c, body, (body.message or "").strip()))


# ── approvals ────────────────────────────────────────────────────────────────

@router.post("/approve", summary="Allow / Not now for something Lucille wanted to write")
async def approve(body: ApproveIn, c: Caller = Depends(caller)):
    require_consent(c, "ai_reflection")
    rec = tools_mod.pending(c.repo, c.uid, body.id)
    if not rec:
        raise HTTPException(404, {"code": "approval_not_found",
                                  "message": "That request has expired. Nothing was written."})
    if not body.allow:
        tools_mod.settle(c.repo, c.uid, body.id, "declined")
        return {"id": body.id, "allowed": False, "events": [], "written": False}
    tool = tools_mod.get(rec["tool"])
    if tool is None:
        tools_mod.settle(c.repo, c.uid, body.id, "failed")
        raise HTTPException(400, {"code": "unknown_tool", "message": "Lucille can't do that any more."})
    if tool.needs_personalization and not has_consent(c, "personalization"):
        tools_mod.settle(c.repo, c.uid, body.id, "blocked")
        raise HTTPException(403, {"code": "consent_required", "purpose": "personalization",
                                  "action": "PUT /v1/privacy/consents"})
    outcome = await tools_mod.run(_tool_ctx(c, rec.get("sessionId") or ""), rec["tool"], rec.get("input") or {})
    failed = bool(outcome.result.get("error"))
    tools_mod.settle(c.repo, c.uid, body.id, "failed" if failed else "allowed")
    events: List[Dict[str, Any]] = []
    if outcome.status:
        events.append(_status(outcome.status["kind"], outcome.status["label"]))
    events.extend(outcome.events)
    return {"id": body.id, "allowed": True, "written": not failed, "events": events,
            "result": outcome.result}


# ── threads ──────────────────────────────────────────────────────────────────

@router.get("/threads", summary="The person's Lucille threads, newest first")
def threads(c: Caller = Depends(caller), limit: int = 50):
    return {"items": ctx_mod.list_threads(c.repo, c.uid, limit=max(1, min(100, limit)))}


@router.get("/threads/{session_id}", summary="One thread's messages")
def thread(session_id: str, c: Caller = Depends(caller)):
    doc = ctx_mod.thread_doc(c.repo, c.uid, session_id)
    if not doc or doc.get("deletedAt"):
        raise HTTPException(404, "Thread not found")
    return {"sessionId": session_id, "title": doc.get("title"), "createdAt": doc.get("createdAt"),
            "updatedAt": doc.get("updatedAt"), "summary": doc.get("summary"),
            "messages": ctx_mod.thread_messages(c.repo, c.uid, session_id)}


@router.delete("/threads/{session_id}", status_code=204, summary="Delete a thread and its shared session")
def delete_thread(session_id: str, c: Caller = Depends(caller)):
    doc = ctx_mod.thread_doc(c.repo, c.uid, session_id)
    if not doc:
        raise HTTPException(404, "Thread not found")
    ctx_mod.delete_thread(c.repo, c.uid, session_id)
    return None


# ─── morning brief ───────────────────────────────────────────────────────────

_BRIEF_SYSTEM = (
    "You are Lucille's internal brief writer. Given a person's recent Escape data, "
    "produce a short morning brief as JSON. Be warm but concise, in Lucille's voice "
    "(warm, plain words, never diagnostic, never over-promising). Only mention what "
    "the data actually says — never invent observations.\n\n"
    "Return JSON with:\n"
    '  "greeting": one short sentence (use their name if you have it),\n'
    '  "noticed": array of 1-3 short observations from the data (skip if nothing to say),\n'
    '  "suggestion": {"title": short action name, "why": one clause, '
    '"route": a v6 deeplink or null}\n\n'
    "Valid routes: v6CheckIn, v6Mind, v6Reset, v6Body, v6DailyPlan, v6Progress, "
    "v6Exercise?id=<id>, v6Plan?planId=<id>.\n"
    "If the data is empty, return only a greeting. No markdown, no explanation — pure JSON."
)


@router.get("/brief", summary="Lucille's morning brief for the Home card")
async def brief(c: Caller = Depends(caller)):
    if not has_consent(c, "ai_reflection"):
        return {"brief": None}
    tz = c.tz or "UTC"
    snap = ctx_mod.escape_snapshot(c.repo, c.uid, tz)
    snap_str = ctx_mod.snapshot_text(snap)

    profile = ctx_mod.profile_text(c.repo, c.uid, tz)
    mems_str, _ = ctx_mod.memories_text(c.uid)

    prompt_parts = [profile]
    if mems_str:
        prompt_parts.append(f"Memories of past conversations:\n{mems_str}")
    prompt_parts.append(f"Current state:\n{snap_str}")
    user_msg = "\n\n".join(prompt_parts)

    if not claude.available():
        return {"brief": None}
    try:
        text, _ = await claude.get_client().text(
            _BRIEF_SYSTEM,
            [{"role": "user", "content": user_msg}],
            model=claude.MODEL_FAST,
            max_tokens=600,
        )
        parsed = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
        return {"brief": parsed}
    except Exception as e:
        logger.warning("brief generation failed for %s: %s", c.uid, e)
        return {"brief": None}
