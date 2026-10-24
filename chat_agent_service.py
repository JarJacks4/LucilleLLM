"""OpenAI tool interface for Lucille's main chat and streaming endpoints.

The main API imports this module at startup. Keep it side-effect free: the older
standalone LangChain agent eagerly loaded HuggingFace models, FAISS pickles, and
OpenAI clients while importing, and relied on APIs removed in newer LangChain.
The running app owns its configured FAISS retriever and OpenAI client instead.
"""

from functools import lru_cache
from typing import Any, Callable, Mapping


class ChatAgentService:
    """Provide the small tool contract used by main.py's OpenAI chat loops."""

    _TOOL_NAME = "selfcare_search"

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return valid OpenAI chat-completions function tool specifications."""
        return [
            {
                "type": "function",
                "function": {
                    "name": self._TOOL_NAME,
                    "description": (
                        "Search the trusted self-care knowledge base for relevant "
                        "wellbeing information before answering a question."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Self-care question or topic to research.",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def execute_tool_call(
        self,
        *,
        tool_name: str,
        tool_args: Mapping[str, Any],
        user_id: str | None,
        retrieve_context_fn: Callable[[str], str],
    ) -> str:
        """Search the existing RAG retriever without loading models at import.

        User identity is accepted for interface compatibility, but the tool only
        searches the public knowledge base; it never reads user-specific data.
        """
        if tool_name != self._TOOL_NAME:
            return "Tool unavailable."
        query = tool_args.get("query") if isinstance(tool_args, Mapping) else None
        if not isinstance(query, str) or not query.strip():
            return "A non-empty search query is required."
        try:
            context = retrieve_context_fn(query.strip()[:1000])
        except Exception:
            return "Knowledge-base search is temporarily unavailable."
        return str(context)[:12000] if context else "No matching knowledge-base content found."


@lru_cache(maxsize=1)
def get_chat_agent_service() -> ChatAgentService:
    """Return the stateless tool adapter used by both chat endpoints."""
    return ChatAgentService()
