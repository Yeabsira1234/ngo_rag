import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when application configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Typed application settings loaded from environment variables."""

    openai_api_key: str = field(repr=False)
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4.1-mini"
    document_path: Path = Path("data/InternationalHandbook.pdf")
    document_source_name: str = "InternationalHandbook.pdf"
    chunk_size: int = 800
    chunk_overlap: int = 150
    chroma_collection_name: str = "ngo_documents"
    chroma_persist_directory: Path = Path("chroma_data")
    retrieval_result_count: int = 4

    def __post_init__(self) -> None:
        if not self.openai_api_key.strip():
            raise ConfigurationError("OPENAI_API_KEY must not be empty.")
        if not self.embedding_model.strip():
            raise ConfigurationError("OPENAI_EMBEDDING_MODEL must not be empty.")
        if not self.llm_model.strip():
            raise ConfigurationError("OPENAI_LLM_MODEL must not be empty.")
        if not self.document_source_name.strip():
            raise ConfigurationError("DOCUMENT_SOURCE_NAME must not be empty.")
        if not self.chroma_collection_name.strip():
            raise ConfigurationError("CHROMA_COLLECTION_NAME must not be empty.")
        if self.chunk_size <= 0:
            raise ConfigurationError("CHUNK_SIZE must be greater than zero.")
        if self.chunk_overlap < 0:
            raise ConfigurationError("CHUNK_OVERLAP cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )
        if self.retrieval_result_count <= 0:
            raise ConfigurationError(
                "RETRIEVAL_RESULT_COUNT must be greater than zero."
            )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "Settings":
        """Build settings from a mapping or the process environment."""
        if env is None:
            load_dotenv()
            env = os.environ

        api_key = env.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY was not found in the environment or .env file."
            )

        return cls(
            openai_api_key=api_key,
            embedding_model=env.get(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            llm_model=env.get("OPENAI_LLM_MODEL", "gpt-4.1-mini"),
            document_path=Path(
                env.get("DOCUMENT_PATH", "data/InternationalHandbook.pdf")
            ),
            document_source_name=env.get(
                "DOCUMENT_SOURCE_NAME", "InternationalHandbook.pdf"
            ),
            chunk_size=_read_int(env, "CHUNK_SIZE", 800),
            chunk_overlap=_read_int(env, "CHUNK_OVERLAP", 150),
            chroma_collection_name=env.get(
                "CHROMA_COLLECTION_NAME", "ngo_documents"
            ),
            chroma_persist_directory=Path(
                env.get("CHROMA_PERSIST_DIRECTORY", "chroma_data")
            ),
            retrieval_result_count=_read_int(
                env, "RETRIEVAL_RESULT_COUNT", 4
            ),
        )


def _read_int(
    env: Mapping[str, str],
    variable_name: str,
    default: int,
) -> int:
    raw_value = env.get(variable_name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as error:
        raise ConfigurationError(
            f"{variable_name} must be a valid integer."
        ) from error
