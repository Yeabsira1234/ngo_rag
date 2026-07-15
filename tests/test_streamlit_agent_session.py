from unittest.mock import Mock

import pytest

import streamlit_app
from src.agent.models import AgentCitation, AgentResponse, AgentStatus
from src.config import Settings


def response(*, status=AgentStatus.DIRECT_ANSWER, tools=(), citations=()):
    return AgentResponse(
        answer="Agent answer",
        status=status,
        citations=citations,
        document_tool_used="document_search" in tools,
        tool_sources=tools,
    )


def test_same_session_preserves_agent_and_separate_sessions_are_isolated(monkeypatch) -> None:
    settings = Settings(openai_api_key="test")
    first_agent, second_agent = Mock(), Mock()
    factory = Mock(side_effect=[first_agent, second_agent])
    monkeypatch.setattr(streamlit_app, "build_agent_service", factory)
    first_state, second_state = {}, {}
    assert streamlit_app.get_session_agent(settings, first_state) is first_agent
    assert streamlit_app.get_session_agent(settings, first_state) is first_agent
    assert streamlit_app.get_session_agent(settings, second_state) is second_agent
    assert factory.call_count == 2


def test_chat_questions_call_agent_and_preserve_follow_up_instance() -> None:
    agent = Mock()
    agent.answer.side_effect = [response(), response()]
    messages = []
    streamlit_app.submit_agent_question(messages, agent, "First question")
    streamlit_app.submit_agent_question(messages, agent, "What was my previous question?")
    assert agent.answer.call_args_list[0].args == ("First question",)
    assert agent.answer.call_args_list[1].args == ("What was my previous question?",)
    assert len(messages) == 4


def test_clear_history_clears_visible_and_internal_memory() -> None:
    agent = Mock()
    state = {streamlit_app.MESSAGES_KEY: [Mock()]}
    streamlit_app.clear_session_history(agent, state)
    assert state[streamlit_app.MESSAGES_KEY] == []
    agent.clear_memory.assert_called_once_with()


def test_failed_turn_adds_complete_safe_visible_exchange() -> None:
    agent = Mock()
    agent.answer.side_effect = RuntimeError("api key and private path")
    messages = []
    streamlit_app.submit_agent_question(messages, agent, "Question")
    assert [message.role for message in messages] == ["user", "assistant"]
    assert "private path" not in messages[-1].content


@pytest.mark.parametrize(
    ("status", "tool"),
    [
        (AgentStatus.ORGANIZATION_ANSWER, "organization_info"),
        (AgentStatus.SQL_ANSWER, "sql_query"),
    ],
)
def test_tool_provenance_is_stored_without_internal_payloads(status, tool) -> None:
    agent = Mock()
    agent.answer.return_value = response(status=status, tools=(tool,))
    messages = []
    streamlit_app.submit_agent_question(messages, agent, "Question")
    assistant = messages[-1]
    assert assistant.tools_used == (tool,)
    assert assistant.agent_status == status.value
    assert "SELECT" not in assistant.content


def test_document_citations_are_converted_for_visible_history() -> None:
    citation = AgentCitation("guide.pdf", 2, 3, 0.4, "uploads/guide.pdf", "id")
    agent = Mock()
    agent.answer.return_value = response(
        status=AgentStatus.DOCUMENT_ANSWER,
        tools=("document_search",),
        citations=(citation,),
    )
    messages = []
    streamlit_app.submit_agent_question(messages, agent, "Document question")
    assert messages[-1].citations[0].source_relative_path == "uploads/guide.pdf"
