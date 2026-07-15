from unittest.mock import Mock

import agent_chat
from src.agent.models import AgentResponse, AgentStatus
from src.config import Settings


def test_agent_cli_displays_direct_answer(monkeypatch, capsys) -> None:
    settings = Settings(openai_api_key="test-key")
    agent = Mock()
    agent.answer.return_value = AgentResponse(
        answer="Hello!",
        status=AgentStatus.DIRECT_ANSWER,
        citations=(),
        document_tool_used=False,
    )
    inputs = iter(["Say hello", "exit"])
    monkeypatch.setattr(agent_chat.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(agent_chat, "build_agent_service", lambda _: agent)
    monkeypatch.setattr(
        agent_chat, "configure_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    exit_code = agent_chat.run()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Answer:\nHello!" in output


def test_agent_cli_displays_structured_tool_source(monkeypatch, capsys) -> None:
    settings = Settings(openai_api_key="test-key")
    agent = Mock()
    agent.answer.return_value = AgentResponse(
        answer="Community Support Network",
        status=AgentStatus.ORGANIZATION_ANSWER,
        citations=(),
        document_tool_used=False,
        tool_sources=("organization_info",),
    )
    inputs = iter(["What is the organization name?", "exit"])
    monkeypatch.setattr(agent_chat.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(agent_chat, "build_agent_service", lambda _: agent)
    monkeypatch.setattr(
        agent_chat, "configure_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    exit_code = agent_chat.run()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Tools used: organization_info" in output


def test_agent_cli_hides_dependency_details_and_continues(monkeypatch, capsys) -> None:
    settings = Settings(openai_api_key="test-key")
    agent = Mock()
    agent.answer.side_effect = RuntimeError("secret provider detail")
    inputs = iter(["A question", "exit"])
    monkeypatch.setattr(agent_chat.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(agent_chat, "build_agent_service", lambda _: agent)
    monkeypatch.setattr(
        agent_chat, "configure_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    exit_code = agent_chat.run()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert agent_chat.REQUEST_ERROR_MESSAGE in output
    assert "secret provider detail" not in output


def test_agent_cli_clear_command_removes_history(monkeypatch, capsys) -> None:
    settings = Settings(openai_api_key="test-key")
    agent = Mock()
    inputs = iter(["/clear", "exit"])
    monkeypatch.setattr(agent_chat.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(agent_chat, "build_agent_service", lambda _: agent)
    monkeypatch.setattr(
        agent_chat, "configure_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    exit_code = agent_chat.run()
    output = capsys.readouterr().out

    assert exit_code == 0
    agent.clear_memory.assert_called_once_with()
    agent.answer.assert_not_called()
    assert "Conversation history cleared." in output
    assert "lasts only until this process exits" in output
