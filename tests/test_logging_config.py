import logging
from pathlib import Path

from src.logging_config import configure_logging


def test_configure_logging_sets_requested_level(tmp_path: Path) -> None:
    log_path = tmp_path / "application.log"

    configure_logging("DEBUG", log_path=log_path)

    assert logging.getLogger().level == logging.DEBUG


def test_logging_redacts_api_key_from_messages_and_exceptions(
    tmp_path: Path,
) -> None:
    api_key = "sk-sensitive-test-key"
    log_path = tmp_path / "application.log"
    configure_logging(
        "INFO",
        sensitive_values=(api_key,),
        log_path=log_path,
    )
    logger = logging.getLogger("test.redaction")

    try:
        raise RuntimeError(f"dependency rejected {api_key}")
    except RuntimeError:
        logger.exception("request failed with key=%s", api_key)

    for handler in logging.getLogger().handlers:
        handler.flush()
    log_contents = log_path.read_text(encoding="utf-8")

    assert api_key not in log_contents
    assert "[REDACTED]" in log_contents
    assert "Traceback" in log_contents
