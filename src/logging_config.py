import logging
from collections.abc import Iterable
from pathlib import Path


class RedactingFormatter(logging.Formatter):
    """Format log records while removing configured sensitive values."""

    def __init__(
        self,
        format_string: str,
        sensitive_values: Iterable[str] = (),
    ) -> None:
        super().__init__(format_string)
        self.sensitive_values = tuple(
            value for value in sensitive_values if value
        )

    def format(self, record: logging.LogRecord) -> str:
        formatted_record = super().format(record)
        for sensitive_value in self.sensitive_values:
            formatted_record = formatted_record.replace(
                sensitive_value,
                "[REDACTED]",
            )
        return formatted_record


def configure_logging(
    log_level: str,
    *,
    sensitive_values: Iterable[str] = (),
    log_path: str | Path = "logs/application.log",
) -> None:
    """Configure the application's single file-based logging pipeline."""
    numeric_level = logging.getLevelNamesMapping().get(log_level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {log_level}")

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    for handler in list(root_logger.handlers):
        if getattr(handler, "_ngo_rag_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(numeric_level)
    handler.setFormatter(
        RedactingFormatter(
            (
                "%(asctime)s level=%(levelname)s logger=%(name)s "
                "%(message)s"
            ),
            sensitive_values=sensitive_values,
        )
    )
    handler._ngo_rag_handler = True  # type: ignore[attr-defined]
    root_logger.addHandler(handler)

    for noisy_logger_name in ("chromadb", "httpx", "openai"):
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)
