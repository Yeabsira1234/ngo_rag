import logging
import time

from src.application import build_application_tools
from src.config import Settings
from src.mcp.adapters import MCPAdapters
from src.mcp.server import create_mcp_server
from src.mcp.utilities import configure_mcp_logging

logger = logging.getLogger(__name__)


def build_server(settings: Settings):
    adapters = MCPAdapters.from_tools(
        build_application_tools(settings),
        max_input_length=settings.mcp_max_input_length,
    )
    return create_mcp_server(adapters)


def run() -> int:
    started = time.monotonic()
    settings: Settings | None = None
    try:
        settings = Settings.from_env()
        configure_mcp_logging(
            settings.log_level,
            sensitive_values=(settings.openai_api_key,),
        )
        server = build_server(settings)
        server.run(transport="stdio")
        return 0
    except Exception:
        sensitive = (settings.openai_api_key,) if settings is not None else ()
        configure_mcp_logging("ERROR", sensitive_values=sensitive)
        logger.error(
            "event=mcp_tool_completed tool=server_startup duration_ms=%d "
            "status=error failure_category=startup_failure",
            round((time.monotonic() - started) * 1000),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
