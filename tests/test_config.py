from pathlib import Path

import pytest

from src.config import ConfigurationError, Settings


def test_settings_use_current_application_defaults() -> None:
    settings = Settings.from_env({"OPENAI_API_KEY": "test-key"})

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.llm_model == "gpt-4.1-mini"
    assert settings.documents_directory == Path("data/samples")
    assert settings.document_glob == "*.pdf"
    assert settings.upload_directory == Path("data/uploads")
    assert settings.max_upload_file_size_mb == 10
    assert settings.max_upload_files == 10
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 150
    assert settings.chroma_collection_name == "ngo_documents"
    assert settings.chroma_persist_directory == Path("chroma_data")
    assert settings.retrieval_result_count == 4
    assert settings.retrieval_max_distance == 0.9
    assert settings.agent_max_tool_iterations == 2
    assert settings.agent_max_tool_calls_per_turn == 3
    assert settings.agent_memory_max_turns == 10
    assert settings.external_api_timeout_seconds == 5.0
    assert settings.external_api_max_retries == 2
    assert settings.sql_server == "YEABSIRA"
    assert settings.sql_database == "NGO_RAG"
    assert settings.sql_query_timeout_seconds == 10
    assert settings.sql_max_rows == 100
    assert settings.log_level == "INFO"


def test_settings_read_environment_overrides() -> None:
    settings = Settings.from_env(
        {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_EMBEDDING_MODEL": "embedding-model",
            "OPENAI_LLM_MODEL": "llm-model",
            "DOCUMENTS_DIRECTORY": "data/approved",
            "DOCUMENT_GLOB": "**/*.pdf",
            "CHUNK_SIZE": "1000",
            "CHUNK_OVERLAP": "200",
            "CHROMA_COLLECTION_NAME": "documents",
            "CHROMA_PERSIST_DIRECTORY": "vector_data",
            "RETRIEVAL_RESULT_COUNT": "6",
            "RETRIEVAL_MAX_DISTANCE": "0.75",
            "AGENT_MAX_TOOL_ITERATIONS": "3",
            "AGENT_MAX_TOOL_CALLS_PER_TURN": "2",
            "AGENT_MEMORY_MAX_TURNS": "7",
            "EXTERNAL_API_TIMEOUT_SECONDS": "7.5",
            "EXTERNAL_API_MAX_RETRIES": "4",
            "LOG_LEVEL": "debug",
        }
    )

    assert settings.openai_api_key == "test-key"
    assert settings.embedding_model == "embedding-model"
    assert settings.llm_model == "llm-model"
    assert settings.documents_directory == Path("data/approved")
    assert settings.document_glob == "**/*.pdf"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.chroma_collection_name == "documents"
    assert settings.chroma_persist_directory == Path("vector_data")
    assert settings.retrieval_result_count == 6
    assert settings.retrieval_max_distance == 0.75
    assert settings.agent_max_tool_iterations == 3
    assert settings.agent_max_tool_calls_per_turn == 2
    assert settings.agent_memory_max_turns == 7
    assert settings.external_api_timeout_seconds == 7.5
    assert settings.external_api_max_retries == 4
    assert settings.log_level == "DEBUG"


def test_settings_require_an_api_key() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        Settings.from_env({})


def test_settings_reject_non_integer_values() -> None:
    with pytest.raises(ConfigurationError, match="CHUNK_SIZE"):
        Settings.from_env(
            {
                "OPENAI_API_KEY": "test-key",
                "CHUNK_SIZE": "eight hundred",
            }
        )


def test_settings_reject_overlap_not_smaller_than_chunk_size() -> None:
    with pytest.raises(ConfigurationError, match="CHUNK_OVERLAP"):
        Settings.from_env(
            {
                "OPENAI_API_KEY": "test-key",
                "CHUNK_SIZE": "100",
                "CHUNK_OVERLAP": "100",
            }
        )


def test_api_key_is_not_exposed_in_settings_repr() -> None:
    settings = Settings.from_env({"OPENAI_API_KEY": "super-secret-key"})

    assert "super-secret-key" not in repr(settings)


def test_settings_reject_invalid_retrieval_distance() -> None:
    with pytest.raises(ConfigurationError, match="RETRIEVAL_MAX_DISTANCE"):
        Settings.from_env(
            {
                "OPENAI_API_KEY": "test-key",
                "RETRIEVAL_MAX_DISTANCE": "-0.1",
            }
        )


def test_settings_reject_invalid_log_level() -> None:
    with pytest.raises(ConfigurationError, match="LOG_LEVEL"):
        Settings.from_env(
            {
                "OPENAI_API_KEY": "test-key",
                "LOG_LEVEL": "VERBOSE",
            }
        )


def test_settings_reject_invalid_agent_iteration_limit() -> None:
    with pytest.raises(ConfigurationError, match="AGENT_MAX_TOOL_ITERATIONS"):
        Settings.from_env(
            {
                "OPENAI_API_KEY": "test-key",
                "AGENT_MAX_TOOL_ITERATIONS": "0",
            }
        )


def test_settings_reject_invalid_agent_memory_limit() -> None:
    with pytest.raises(ConfigurationError, match="AGENT_MEMORY_MAX_TURNS"):
        Settings.from_env(
            {
                "OPENAI_API_KEY": "test-key",
                "AGENT_MEMORY_MAX_TURNS": "0",
            }
        )


def test_settings_reject_invalid_agent_tool_call_limit() -> None:
    with pytest.raises(ConfigurationError, match="AGENT_MAX_TOOL_CALLS_PER_TURN"):
        Settings.from_env(
            {
                "OPENAI_API_KEY": "test-key",
                "AGENT_MAX_TOOL_CALLS_PER_TURN": "0",
            }
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EXTERNAL_API_TIMEOUT_SECONDS", "0"),
        ("EXTERNAL_API_TIMEOUT_SECONDS", "nan"),
        ("EXTERNAL_API_MAX_RETRIES", "-1"),
    ],
)
def test_settings_reject_invalid_external_api_limits(name: str, value: str) -> None:
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env({"OPENAI_API_KEY": "test-key", name: value})
