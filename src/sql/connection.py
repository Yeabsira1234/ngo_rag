from collections.abc import Callable
from typing import Any

from src.config import Settings
from src.sql.repository import Connection, SQLConfigurationError


def build_connection_factory(settings: Settings) -> Callable[[], Connection]:
    parts = {
        "DRIVER": "{" + settings.sql_driver + "}",
        "SERVER": settings.sql_server,
        "DATABASE": settings.sql_database,
        "Trusted_Connection": "yes",
        "TrustServerCertificate": (
            "yes" if settings.sql_trust_server_certificate else "no"
        ),
        "ApplicationIntent": "ReadOnly",
    }
    connection_string = ";".join(f"{key}={value}" for key, value in parts.items())

    def connect() -> Any:
        if not settings.sql_trusted_connection:
            raise SQLConfigurationError(
                "Only Windows Authentication is supported in this step."
            )
        import pyodbc
        return pyodbc.connect(
            connection_string,
            timeout=settings.sql_query_timeout_seconds,
        )

    return connect
