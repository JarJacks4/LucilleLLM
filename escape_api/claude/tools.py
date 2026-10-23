"""
The tools Claude may call during a Lucille turn.

Each tool declares:
  * `spec`     — the Anthropic tool definition sent with the request
  * `kind`     — which `status.kind` it reports (reading / drafting / writing / checking)
  * `label`    — the past-tense status line emitted AFTER it ran, describing what
                 really happened ("Read your last 3 check-ins", "Made a PDF").
                 A status is never emitted for something that did not happen.
  * `writes`   — True when it changes something outside the conversation. Those are
                 never executed inline: the turn emits an `approval` event and the
                 app has to call POST /v2/lucille/approve. Nothing is written on a
                 timeout (approvals expire).

Reads come from the v1 repo and the shared services; nothing here invents data.
Routes are always validated through core.resolve_route, so a chip can never point
at a page that doesn't exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional

from escape_api import core, documents, plans_claude, selfcare
from escape_api.repo import Repo, user_path
from escape_api.settings import cfg

logger = logging.getLogger(__name__)

APPROVALS = "lucille_approvals"
APPROVAL_TTL_MINUTES = 15


# ── context + outcome ────────────────────────────────────────────────────────

@dataclass
class ToolCtx:
    repo: Repo
    uid: str
    tz: str
    session_id: str
    has_personalization: bool = True
    quota_ok: Callable[[str, int], bool] = lambda kind, limit: True


@dataclass
class ToolOutcome:
    result: Dict[str, Any]                                  # tool_result content for Claude
    events: List[Dict[str, Any]] = field(default_factory=list)   # extra SSE events
    status: Optional[Dict[str, str]] = None                 # the completed status line


def _status(kind: str, label: str) -> Dict[str, str]:
    return {"kind": kind, "label": label}


def _public(path: str) -> str:
    return f"{cfg().public_base_url.rstrip('/')}{path}"


# ── reads ────────────────────────────────────────────────────────────────────

async def read_mood_summary(ctx: ToolCtx, args: Dict[str, Any]) -> ToolOutcome:
    rng = args.get("range") if args.get("range") in documents.RANGES else "30d"
    n = documents.RANGES[rng]
    since = (core.now_utc() - timedelta(days=n)).date().isoformat()
    rows = [d for _, d in ctx.repo.query(user_path(ctx.uid, "mood_checkins"),
                                         order_by="createdAt", desc=True, limit=40)]
    recent = [r for r in rows if (r.get("localDay") or "") >= since]
    days = [d for _, d in ctx.repo.query(user_path(ctx.uid, "mood_days"), where=[("day", ">=", since)])]
    count = sum(int(d.get("count") or 0) for d in days) or len(recent)
    avg = None
    if days and count:
        total = sum(float(d.get("sumV") or 0) for d in days)
        avg = round(100 * total / sum(int(d.get("count") or 0) for d in days))
    words: Dict[str, int] = {}
    for r in recent:
        if r.get("word"):
            words[r["word"]] = words.get(r["word"], 0) + 1
    result = {
        "range": rng, "checkins": count,
        "averageMood": avg,
        "mostCommonWords": sorted(words.items(), key=lambda kv: -kv[1])[:4],
        "latest": ({"word": recent[0].get("word"), "tone": recent[0].get("tone"),
                    "day": recent[0].get("localDay")} if recent else None),
        "empty": count == 0,
        "note": "No mood check-ins in that window." if count == 0 else "",
    }
    label = (f"Read your last {min(len(recent), 3)} check-in{'s' if min(len(recent), 3) != 1 else ''}"
             if recent else "Looked for check-ins (none yet)")
    return ToolOutcome(result, status=_status("reading", label))


async def read_journal_summary(ctx: ToolCtx, args: Dict[str, Any]) -> ToolOutcome:
    days = int(args.get("days") or 14)
    days = max(7, min(90, days))
    since = (core.now_utc() - timedelta(days=days)).date().isoformat()
    rows = [d for _, d in ctx.repo.query(user_path(ctx.uid, "journal_entries"),
                                         where=[("localDay", ">=", since)])
            if d.get("status") == "saved"]
    themes: Dict[str, int] = {}
    tones: Dict[str, int] = {}
    for e in rows:
        for t in e.get("themes") or []:
            themes[t] = themes.get(t, 0) + 1
        tone = (e.get("mood") or {}).get("tone")
        if tone:
            tones[tone] = tones.get(tone, 0) + 1
    result = {
        "days": days, "entries": len(rows),
        "activeDays": len({e.get("localDay") for e in rows if e.get("localDay")}),
        "topThemes": sorted(themes.items(), key=lambda kv: -kv[1])[:5],
        "tones": sorted(tones.items(), key=lambda kv: -kv[1])[:4],
        "lastEntryDay": max((e.get("localDay") or "" for e in rows), default=None) or None,
        "empty": not rows,
        "note": "Nothing written in the journal in that window." if not rows else "",
        "privacy": "Entry text is not shared with you; only counts, themes and tone.",
    }
    label = f"Read the tone of {len(rows)} journal entr{'y' if len(rows) == 1 else 'ies'}" if rows \
        else "Looked at your journal (nothing written yet)"
    return ToolOutcome(result, status=_status("reading", label))


async def read_plan(ctx: ToolCtx, args: Dict[str, Any]) -> ToolOutcome:
    plan = plans_claude.active_plan(ctx.repo, ctx.uid)
    if not plan:
        return ToolOutcome({"plan": None, "empty": True, "note": "No active plan."},
                           status=_status("reading", "Checked your plan (none active)"))
    today = plans_claude.today_items(plan, ctx.tz)
    result = {
        "plan": {"id": plan["id"], "title": plan.get("title"), "days": plan.get("days"),
                 "minutesPerDay": plan.get("minutesPerDay"), "provider": plan.get("provider"),
                 "revision": plan.get("revision"), "why": plan.get("why"),
                 "goals": plan.get("goalLabels") or plan.get("goals")},
        "today": [{"id": i["id"], "title": i["title"], "minutes": i["minutes"], "done": bool(i.get("done"))}
                  for i in today],
        "doneTotal": sum(1 for i in plan.get("items", []) if i.get("done")),
        "itemsTotal": len(plan.get("items", [])),
        "empty": False,
    }
    return ToolOutcome(result, status=_status("reading", "Read your plan"))


async def read_progress(ctx: ToolCtx, args: Dict[str, Any]) -> ToolOutcome:
    st = core.get_state(ctx.repo, ctx.uid) or {}
    score = selfcare.compute_score(ctx.repo, ctx.uid, ctx.tz)
    result = {
        "score": score.get("score"), "band": score.get("band"), "delta": score.get("delta"),
        "components": {k: v.get("detail") for k, v in (score.get("components") or {}).items()},
        "streakDays": int(st.get("streakDays") or 0), "bestStreak": int(st.get("bestStreak") or 0),
        "coins": int(st.get("coins") or 0),
        "disclaimer": score.get("disclaimer"),
    }
    return ToolOutcome(result, status=_status("reading", f"Read your score ({score.get('score')})"))


# ── writes (approval required) ───────────────────────────────────────────────

async def draft_plan(ctx: ToolCtx, args: Dict[str, Any]) -> ToolOutcome:
    goals = [str(g)[:60] for g in (args.get("goals") or [])][:3]
    minutes = int(args.get("minutesPerDay") or 15)
    days = int(args.get("days") or 7)
    notes = args.get("notes")
    use_claude = ctx.has_personalization and ctx.quota_ok("claude_plan", cfg().quota_claude_plan_per_day)
    plan, statuses = await plans_claude.author_plan(ctx.repo, ctx.uid, ctx.tz, goals, minutes, days,
                                                    notes, use_claude=use_claude)
    events = [{"type": "plan", "planId": plan["id"], "title": plan.get("title"), "days": plan.get("days"),
               "minutesPerDay": plan.get("minutesPerDay"), "updated": plan.get("createdAt")}]
    result = {"planId": plan["id"], "title": plan.get("title"), "days": plan.get("days"),
              "minutesPerDay": plan.get("minutesPerDay"), "provider": plan.get("provider"),
              "items": [{"day": i["day"], "title": i["title"], "minutes": i["minutes"]} for i in plan["items"]],
              "why": plan.get("why")}
    label = statuses[-1]["label"] if statuses else "Drafted your week"
    return ToolOutcome(result, events=events, status=_status("drafting", label))


async def adjust_plan(ctx: ToolCtx, args: Dict[str, Any]) -> ToolOutcome:
    plan = plans_claude.active_plan(ctx.repo, ctx.uid)
    if not plan:
        return ToolOutcome({"error": "no_active_plan", "note": "There is no active plan to adjust."},
                           status=_status("checking", "Checked your plan (none active)"))
    adjust = args.get("adjust")
    if adjust not in plans_claude.ADJUSTMENTS:
        return ToolOutcome({"error": "bad_adjustment",
                            "note": f"adjust must be one of {list(plans_claude.ADJUSTMENTS)}"})
    plan, changed = await plans_claude.adjust_plan(ctx.repo, ctx.uid, ctx.tz, plan, adjust, args.get("note"))
    events = [{"type": "plan", "planId": plan["id"], "title": plan.get("title"), "days": plan.get("days"),
               "minutesPerDay": plan.get("minutesPerDay"), "updated": plan.get("updatedAt")}]
    return ToolOutcome({"planId": plan["id"], "changed": changed, "revision": plan.get("revision"),
                        "why": plan.get("why")},
                       events=events,
                       status=_status("drafting", f"Adjusted your plan ({len(changed)} change"
                                                  f"{'s' if len(changed) != 1 else ''})"))


async def make_plan_document(ctx: ToolCtx, args: Dict[str, Any]) -> ToolOutcome:
    fmt = args.get("format") if args.get("format") in ("pdf", "md", "docx") else "pdf"
    include = [i for i in (args.get("include") or ["schedule", "progress", "notes"])
               if i in ("schedule", "progress", "notes")] or ["schedule"]
    plan = plans_claude.active_plan(ctx.repo, ctx.uid)
    if not plan:
        return ToolOutcome({"error": "no_active_plan", "note": "There is no plan to turn into a document."},
                           status=_status("checking", "Checked your plan (none active)"))
    if not ctx.quota_ok("claude_doc", cfg().quota_claude_doc_per_day):
        return ToolOutcome({"error": "quota_exceeded",
                            "note": "They have made as many documents as the daily limit allows."})
    charts = [documents.series(ctx.repo, ctx.uid, ctx.tz, m, "30d", plan) for m in ("adherence", "mood")] \
        if "progress" in include else []
    st = core.get_state(ctx.repo, ctx.uid) or {}
    score = selfcare.compute_score(ctx.repo, ctx.uid, ctx.tz)
    progress = {"score": score.get("score"), "band": score.get("band"),
                "streakDays": int(st.get("streakDays") or 0)}
    try:
        data, mime = documents.render(fmt, plan, include, charts, progress)
    except documents.DocumentUnavailable as e:
        return ToolOutcome({"error": e.code, "note": str(e)})
    rec = documents.store_document(ctx.repo, ctx.uid, plan, fmt, data, mime, ctx.tz)
    events = [{"type": "artifact", "id": rec["id"], "kind": "plan_document", "title": rec["filename"],
               "url": rec["url"], "mime": rec["mime"], "bytes": rec["bytes"]}]
    return ToolOutcome({"documentId": rec["id"], "filename": rec["filename"], "bytes": rec["bytes"],
                        "expiresAt": rec["expiresAt"],
                        "charts": [{"metric": c["metric"], "empty": c["empty"]} for c in charts]},
                       events=events,
                       status=_status("writing", f"Made a {fmt.upper()}"))


# ── chart (reads only, so no approval) ───────────────────────────────────────

async def make_chart(ctx: ToolCtx, args: Dict[str, Any]) -> ToolOutcome:
    metric = args.get("metric") if args.get("metric") in documents.METRICS else "mood"
    rng = args.get("range") if args.get("range") in documents.RANGES else "30d"
    plan = plans_claude.active_plan(ctx.repo, ctx.uid)
    data = documents.series(ctx.repo, ctx.uid, ctx.tz, metric, rng, plan)
    svg = documents.chart_svg(data)
    url = _public(f"/v2/plans/{(plan or {}).get('id', 'none')}/chart?metric={metric}&range={rng}")
    events = [{"type": "artifact", "id": core.new_id("chart"), "kind": "chart",
               "title": f"{data['title']} · last {rng}", "url": url, "mime": documents.MIME["svg"],
               "bytes": len(svg.encode("utf-8")), "svg": svg,
               "series": data["points"], "empty": data["empty"]}]
    result = {"metric": metric, "range": rng, "empty": data["empty"], "note": data["note"],
              "average": data["average"], "max": data["max"],
              "points": [p for p in data["points"] if p["value"] is not None][-14:]}
    label = (f"Made a {data['title'].lower()} chart" if not data["empty"]
             else f"Checked your {metric} history (not enough yet)")
    return ToolOutcome(result, events=events, status=_status("writing", label))


# ── route suggestion ─────────────────────────────────────────────────────────

async def suggest_route(ctx: ToolCtx, args: Dict[str, Any]) -> ToolOutcome:
    name = str(args.get("route") or "")
    params = args.get("params") if isinstance(args.get("params"), dict) else None
    link = core.resolve_route(name, params)
    if not link:
        known = [a["route"] for a in plans_claude.library()[:12]]
        return ToolOutcome({"error": "unknown_route", "note": f"'{name}' is not a page in this app.",
                            "someValidRoutes": known},
                           status=_status("checking", "Checked that page exists"))
    label = str(args.get("label") or "Open this")[:48]
    events = [{"type": "action", "id": core.new_id("act"), "label": label,
               "route": link["route"], "params": link.get("params") or {}, "url": link["url"]}]
    return ToolOutcome({"route": link["route"], "url": link["url"], "label": label},
                       events=events, status=_status("checking", f"Offered to open {label}"))


# ── registry ─────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name: str
    kind: str
    writes: bool
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[ToolCtx, Dict[str, Any]], Awaitable[ToolOutcome]]
    pending_label: str = ""
    approval_label: str = ""
    destructive: bool = False
    needs_personalization: bool = False

    @property
    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def _obj(props: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


_RANGE = {"type": "string", "enum": list(documents.RANGES), "description": "Window to look at."}

TOOLS: Dict[str, Tool] = {t.name: t for t in [
    Tool("read_mood_summary", "reading", False,
         "Read a summary of the person's mood check-ins: how many, their average mood and their most "
         "common words. Counts and words only, never raw text.",
         _obj({"range": _RANGE}), read_mood_summary, pending_label="Reading your check-ins"),
    Tool("read_journal_summary", "reading", False,
         "Read the shape of the person's journal: how many entries, which themes and tones recur. "
         "Entry text is never returned.",
         _obj({"days": {"type": "integer", "minimum": 7, "maximum": 90}}), read_journal_summary,
         pending_label="Reading your journal"),
    Tool("read_plan", "reading", False,
         "Read the person's active self-care plan and what is left for today.",
         _obj({}), read_plan, pending_label="Checking your plan"),
    Tool("read_progress", "reading", False,
         "Read the person's Self-Care Score with its components, their streak and their coins.",
         _obj({}), read_progress, pending_label="Checking your progress"),
    Tool("draft_plan", "drafting", True,
         "Write a new self-care plan from the person's real history. This REPLACES any active plan, so it "
         "needs their permission first.",
         _obj({"goals": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
               "minutesPerDay": {"type": "integer", "minimum": 5, "maximum": 90},
               "days": {"type": "integer", "minimum": 3, "maximum": 14},
               "notes": {"type": "string", "description": "What the person asked for, in their words."}},
              ["goals"]),
         draft_plan, pending_label="Drafting your week",
         approval_label="write you a new plan", needs_personalization=True),
    Tool("adjust_plan", "drafting", True,
         "Change the active plan: 'easier' removes a step a day, 'harder' adds one, 'shift_days' moves "
         "unfinished days forward. Needs their permission.",
         _obj({"adjust": {"type": "string", "enum": list(plans_claude.ADJUSTMENTS)},
               "note": {"type": "string"}}, ["adjust"]),
         adjust_plan, pending_label="Adjusting your plan", approval_label="adjust your plan",
         needs_personalization=True),
    Tool("make_plan_document", "writing", True,
         "Turn the active plan into a file they can keep or share (PDF, Markdown or Word), with the "
         "schedule, their progress and real charts. Needs their permission.",
         _obj({"format": {"type": "string", "enum": ["pdf", "md", "docx"]},
               "include": {"type": "array", "items": {"type": "string",
                                                       "enum": ["schedule", "progress", "notes"]}}}),
         make_plan_document, pending_label="Making your document", approval_label="make a plan document"),
    Tool("make_chart", "writing", False,
         "Draw a chart of the person's own history: plan adherence, minutes of self-care, mood, "
         "Self-Care Score or streak. Says so honestly when there isn't enough history.",
         _obj({"metric": {"type": "string", "enum": list(documents.METRICS)}, "range": _RANGE},
              ["metric"]),
         make_chart, pending_label="Drawing your chart"),
    Tool("suggest_route", "checking", False,
         "Offer a chip that opens a page in the Escape app. The route name must come from a tool result "
         "or the plan; it is validated and dropped if it does not exist.",
         _obj({"route": {"type": "string"}, "label": {"type": "string"},
               "params": {"type": "object"}}, ["route"]),
         suggest_route, pending_label="Checking that page exists"),
]}


def specs(*, allow_writes: bool = True, allow_personalization: bool = True) -> List[Dict[str, Any]]:
    """Tool definitions for this turn. Consent decides what Lucille is even offered."""
    out = []
    for t in TOOLS.values():
        if t.writes and not allow_writes:
            continue
        if t.needs_personalization and not allow_personalization:
            continue
        out.append(t.spec)
    return out


def get(name: str) -> Optional[Tool]:
    return TOOLS.get(name)


async def run(ctx: ToolCtx, name: str, args: Dict[str, Any]) -> ToolOutcome:
    """Execute a tool. Never raises: a failure comes back as a tool_result Claude can read."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolOutcome({"error": "unknown_tool", "note": f"There is no tool called {name}."})
    try:
        return await tool.handler(ctx, args or {})
    except Exception as e:  # noqa: BLE001
        logger.warning("tool %s failed: %s: %s", name, type(e).__name__, e)
        return ToolOutcome({"error": "tool_failed", "note": "That did not work. Tell the person plainly."})


# ── approvals ────────────────────────────────────────────────────────────────

def approval_event(repo: Repo, uid: str, session_id: str, tool: Tool, args: Dict[str, Any]) -> Dict[str, Any]:
    """Record a pending approval and return the `approval` SSE event."""
    aid = core.new_id("apr")
    detail = _approval_detail(tool, args)
    rec = {"id": aid, "sessionId": session_id, "tool": tool.name, "input": args or {},
           "label": f"Lucille wants to {tool.approval_label or tool.name.replace('_', ' ')}",
           "detail": detail, "destructive": bool(tool.destructive),
           "status": "pending", "createdAt": core.iso(core.now_utc()),
           "expiresAt": core.iso(core.now_utc() + timedelta(minutes=APPROVAL_TTL_MINUTES))}
    repo.set(user_path(uid, APPROVALS, aid), rec)
    return {"type": "approval", "id": aid, "label": rec["label"], "detail": detail,
            "destructive": rec["destructive"]}


def _approval_detail(tool: Tool, args: Dict[str, Any]) -> str:
    a = args or {}
    if tool.name == "draft_plan":
        bits = f"{a.get('days', 7)} days, about {a.get('minutesPerDay', 15)} minutes a day"
        goals = ", ".join(str(g) for g in (a.get("goals") or [])[:3])
        return f"A new plan ({bits})" + (f" for {goals}" if goals else "") + ". This replaces your current plan."
    if tool.name == "adjust_plan":
        return {"easier": "Make this week lighter — one step fewer each day.",
                "harder": "Add one more step to each day.",
                "shift_days": "Move everything you haven't done yet forward."}.get(
            a.get("adjust"), "Change your plan.")
    if tool.name == "make_plan_document":
        return (f"A {str(a.get('format', 'pdf')).upper()} of your plan"
                + (" with your progress charts" if "progress" in (a.get("include") or ["progress"]) else "")
                + ", kept for 24 hours.")
    return "Lucille needs your permission for this."


def pending(repo: Repo, uid: str, approval_id: str) -> Optional[Dict[str, Any]]:
    rec = repo.get(user_path(uid, APPROVALS, approval_id))
    if not rec or rec.get("status") != "pending":
        return None
    try:
        if core.now_utc() > core.parse_iso(rec["expiresAt"]):
            repo.set(user_path(uid, APPROVALS, approval_id), {"status": "expired"}, merge=True)
            return None
    except Exception:
        return None
    return rec


def settle(repo: Repo, uid: str, approval_id: str, status: str) -> None:
    repo.set(user_path(uid, APPROVALS, approval_id),
             {"status": status, "settledAt": core.iso(core.now_utc())}, merge=True)
