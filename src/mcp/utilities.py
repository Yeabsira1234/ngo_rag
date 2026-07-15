import logging
import sys
from collections.abc import Iterable
from pathlib import Path

from src.logging_config import RedactingFormatter, configure_logging


def configure_mcp_logging(
    log_level: str,
    *,
    sensitive_values: Iterable[str] = (),
    log_path: str | Path = "logs/application.log",
) -> None:
    """Log to the application file and stderr while reserving stdout for MCP."""
    configure_logging(
        log_level,
        sensitive_values=sensitive_values,
        log_path=log_path,
    )
    numeric_level = logging.getLevelNamesMapping()[log_level.upper()]
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_ngo_mcp_stderr_handler", False):
            root.removeHandler(handler)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(numeric_level)
    stderr_handler.setFormatter(
        RedactingFormatter(
            "%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
            sensitive_values=sensitive_values,
        )
    )
    stderr_handler._ngo_mcp_stderr_handler = True  # type: ignore[attr-defined]
    root.addHandler(stderr_handler)
