"""
Lucille — Claude edition.

Everything in this package talks to the Anthropic Messages API and nothing else:
there is no `openai` import anywhere under escape_api/claude/ (tests/v2 assert it).

  client.py   thin async Messages client (httpx + SSE), retries, 503 when unkeyed
  context.py  "memory since inception" — one turn's system prompt + message window
  persona.py  Lucille's system prompt for this edition
  tools.py    the tools Claude may call, their status events and approval rules
"""

from escape_api.claude import client, context, persona, tools  # noqa: F401

__all__ = ["client", "context", "persona", "tools"]
