"""Regression tests for the main API's chat-agent tool adapter."""

from chat_agent_service import ChatAgentService, get_chat_agent_service


def test_chat_agent_service_is_available_at_import_without_external_models():
    assert isinstance(get_chat_agent_service(), ChatAgentService)
    assert get_chat_agent_service() is get_chat_agent_service()


def test_chat_tool_definitions_match_openai_function_schema():
    definitions = get_chat_agent_service().get_tool_definitions()
    assert len(definitions) == 1
    assert definitions[0]["type"] == "function"
    function = definitions[0]["function"]
    assert function["name"] == "selfcare_search"
    assert function["parameters"]["required"] == ["query"]


def test_selfcare_search_uses_app_retriever_not_a_second_embedding_model():
    seen = []

    def retrieve(query):
        seen.append(query)
        return "A grounding passage"

    result = get_chat_agent_service().execute_tool_call(
        tool_name="selfcare_search",
        tool_args={"query": "  guided breathing  "},
        user_id="user-123",
        retrieve_context_fn=retrieve,
    )
    assert result == "A grounding passage"
    assert seen == ["guided breathing"]


def test_unrecognized_or_empty_tools_do_not_call_retriever():
    def fail_if_called(query):
        raise AssertionError("retriever should not run")

    svc = get_chat_agent_service()
    for name, args in [("unknown", {"query": "hi"}), ("selfcare_search", {"query": ""})]:
        result = svc.execute_tool_call(
            tool_name=name,
            tool_args=args,
            user_id=None,
            retrieve_context_fn=fail_if_called,
        )
        assert result


def test_retriever_failure_is_handled_without_crashing_chat():
    result = get_chat_agent_service().execute_tool_call(
        tool_name="selfcare_search",
        tool_args={"query": "stress"},
        user_id=None,
        retrieve_context_fn=lambda query: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert "unavailable" in result
