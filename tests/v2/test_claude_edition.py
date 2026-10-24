"""
Edition hygiene and the Anthropic client itself.

The first test is the important one: nothing in the Claude edition may import
`openai`, directly or as a string. Two Lucilles ship side by side and never mix.
"""

import ast
import asyncio
import os

import pytest
from fastapi import FastAPI

from escape_api import register, register_claude
from escape_api.claude import client as claude
from escape_api.settings import reload_settings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLAUDE_EDITION_FILES = [
    "escape_api/claude/__init__.py",
    "escape_api/claude/client.py",
    "escape_api/claude/context.py",
    "escape_api/claude/persona.py",
    "escape_api/claude/tools.py",
    "escape_api/routers/lucille_claude.py",
    "escape_api/routers/plans_v2.py",
    "escape_api/plans_claude.py",
    "escape_api/documents.py",
]


def test_no_file_in_the_claude_edition_imports_openai():
    offenders = []
    for rel in CLAUDE_EDITION_FILES:
        path = os.path.join(ROOT, rel)
        assert os.path.exists(path), f"missing {rel}"
        source = open(path, encoding="utf-8").read()
        tree = ast.parse(source, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in ("openai", "langchain_openai"):
                        offenders.append(f"{rel}: import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in ("openai", "langchain_openai"):
                    offenders.append(f"{rel}: from {node.module} import …")
            elif isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "attr", None) or getattr(fn, "id", None)
                if name in ("import_module", "__import__"):
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and "openai" in str(arg.value):
                            offenders.append(f"{rel}: dynamic import of {arg.value}")
    assert offenders == [], offenders


def test_the_claude_edition_is_off_by_default(monkeypatch):
    monkeypatch.delenv("LUCILLE_EDITION", raising=False)
    reload_settings()
    app = FastAPI()
    assert register_claude(app) is False
    register(app)
    paths = set(app.openapi()["paths"])
    assert not any(p.startswith("/v2/") for p in paths)
    assert "/v1/me" in paths                                      # the v1 app is untouched

    monkeypatch.setenv("LUCILLE_EDITION", "claude")
    reload_settings()
    app2 = FastAPI()
    assert register_claude(app2) is True
    v2 = {p for p in app2.openapi()["paths"] if p.startswith("/v2/")}
    assert {"/v2/lucille/turn", "/v2/lucille/turn/audio", "/v2/lucille/approve",
            "/v2/lucille/threads", "/v2/lucille/threads/{session_id}",
            "/v2/plans", "/v2/plans/active", "/v2/plans/{plan_id}",
            "/v2/plans/{plan_id}/items/{item_id}/complete", "/v2/plans/{plan_id}/document",
            "/v2/plans/{plan_id}/chart"} <= v2


def test_edition_flag_is_reported_in_v1_config(client):
    j = client.get("/v1/config").json()
    assert j["features"]["lucilleEdition"] == "claude"
    assert j["features"]["lucilleClaude"] is True
    assert j["features"]["claudeConfigured"] is False             # no key in tests
    assert j["limits"]["claudeTurnsPerDay"] >= 1


def test_model_ids_match_the_contract():
    assert claude.MODEL_DEFAULT == "claude-opus-5-5"
    assert claude.MODEL_CHEAP == "claude-sonnet-5-5"
    assert claude.MODEL_FAST == "claude-haiku-5-5"
    assert claude.MODEL_LONG == "claude-fable-5-1"
    assert claude.CONTEXT_WINDOW_TOKENS == 1_000_000 and claude.MAX_OUTPUT_TOKENS == 128_000
    reload_settings()
    from escape_api.settings import cfg
    assert cfg().claude_model == claude.MODEL_DEFAULT
    assert cfg().claude_fast_model == claude.MODEL_FAST


def test_client_without_a_key_raises_503_unavailable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    claude.set_client(None)
    assert claude.available() is False
    with pytest.raises(claude.ClaudeUnavailable) as e:
        claude.get_client()
    assert e.value.status_code == 503 and e.value.code == "claude_unavailable"
    assert e.value.as_detail()["code"] == "claude_unavailable"
    # helpers degrade instead of raising
    assert asyncio.run(claude.one_shot("sys", "prompt", fallback="fallback")) == "fallback"


def test_sse_parsing_handles_text_tools_and_usage():
    lines = [
        'event: message_start',
        'data: {"type":"message_start","message":{"usage":{"input_tokens":11,"output_tokens":0}}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi "}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"there"}}',
        'data: {"type":"content_block_start","index":1,'
        '"content_block":{"type":"tool_use","id":"toolu_1","name":"read_plan"}}',
        'data: {"type":"content_block_delta","index":1,'
        '"delta":{"type":"input_json_delta","partial_json":"{\\"a\\":"}}',
        'data: {"type":"content_block_delta","index":1,'
        '"delta":{"type":"input_json_delta","partial_json":"1}"}}',
        'data: {"type":"content_block_stop","index":1}',
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
        '"usage":{"output_tokens":9}}',
        'data: not json',
        ': ping',
    ]

    async def collect():
        async def gen():
            for l in lines:
                yield l
        return [ev async for ev in claude._parse_sse(gen())]

    evs = asyncio.run(collect())
    kinds = [e["type"] for e in evs]
    assert kinds == ["usage", "text", "text", "tool_use", "usage", "stop"]
    assert "".join(e["text"] for e in evs if e["type"] == "text") == "Hi there"
    tool = next(e for e in evs if e["type"] == "tool_use")
    assert tool == {"type": "tool_use", "id": "toolu_1", "name": "read_plan", "input": {"a": 1}}
    assert evs[-1]["reason"] == "tool_use"


def test_retry_delay_respects_retry_after_and_backs_off():
    c = claude.ClaudeClient("k")
    assert c._retry_delay(0, "3") == 3.0
    assert c._retry_delay(0, "nonsense") < 1.0
    assert c._retry_delay(3, None) >= 4.0 and c._retry_delay(9, None) <= 8.3


def test_error_mapping():
    c = claude.ClaudeClient("k")
    for status, code in ((401, "claude_unavailable"), (429, "claude_rate_limited"),
                         (529, "overloaded_error"), (400, "invalid_request_error")):
        with pytest.raises(claude.ClaudeError) as e:
            c._raise_for(status, {"error": {"type": "overloaded_error"
                                            if status == 529 else "invalid_request_error"}})
        assert e.value.code == code or e.value.status_code in (502, 503)


def test_token_estimate_is_monotonic():
    assert claude.estimate_tokens("") >= 1
    assert claude.estimate_tokens("x" * 400) > claude.estimate_tokens("x" * 40)
    est = claude.estimate_messages_tokens("system", [{"role": "user", "content": "hello"}])
    assert est > claude.estimate_tokens("system")


def test_tool_specs_are_valid_anthropic_shapes():
    from escape_api.claude import tools as tools_mod
    for spec in tools_mod.specs():
        assert set(spec) == {"name", "description", "input_schema"}
        assert spec["input_schema"]["type"] == "object"
        assert isinstance(spec["input_schema"]["properties"], dict)
        assert spec["description"] and len(spec["name"]) <= 64
    writes = {name for name, t in tools_mod.TOOLS.items() if t.writes}
    assert writes == {"draft_plan", "adjust_plan", "make_plan_document"}
    assert all(tools_mod.TOOLS[n].approval_label for n in writes)
    kinds = {t.kind for t in tools_mod.TOOLS.values()}
    assert kinds <= {"thinking", "reading", "writing", "searching", "drafting", "checking"}


def test_persona_never_carries_crisis_numbers():
    from escape_api.claude import persona
    text = persona.system_prompt()
    assert "988" not in text and "741741" not in text
    assert "not a therapist" in text and "explain WHY" in text
    assert "let the app supply the crisis" in text
