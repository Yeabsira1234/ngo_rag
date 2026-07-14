from unittest.mock import Mock

import chat
from src.config import ConfigurationError, Settings
from src.rag_service import RAGResponse, RAGStatus


def test_successful_cli_behavior_remains_unchanged(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    settings = Settings(openai_api_key="test-key")
    service = Mock()
    service.answer.return_value = RAGResponse(
        answer="A grounded answer",
        citations=(),
        llm_called=True,
        status=RAGStatus.ANSWERED,
    )
    inputs = iter(["A question", "exit"])
    monkeypatch.setattr(chat.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(chat, "build_rag_service", lambda _: service)
    monkeypatch.setattr(chat, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    exit_code = chat.run()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Document assistant is ready." in output
    assert "Answer:\nA grounded answer" in output
    assert "Goodbye." in output


def test_dependency_failure_is_shown_safely_and_cli_continues(
    monkeypatch,
    capsys,
) -> None:
    settings = Settings(openai_api_key="test-key")
    service = Mock()
    service.answer.side_effect = RuntimeError("private diagnostic detail")
    inputs = iter(["A question", "exit"])
    monkeypatch.setattr(chat.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(chat, "build_rag_service", lambda _: service)
    monkeypatch.setattr(chat, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    exit_code = chat.run()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert chat.REQUEST_ERROR_MESSAGE in captured.out
    assert "private diagnostic detail" not in captured.out
    assert "Traceback" not in captured.out


def test_startup_failure_returns_non_zero_exit_code(
    monkeypatch,
    capsys,
) -> None:
    def raise_configuration_error():
        raise ConfigurationError("invalid configuration")

    monkeypatch.setattr(chat.Settings, "from_env", raise_configuration_error)
    monkeypatch.setattr(chat, "configure_logging", lambda *args, **kwargs: None)

    exit_code = chat.run()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert chat.STARTUP_ERROR_MESSAGE in output
    assert "invalid configuration" not in output
