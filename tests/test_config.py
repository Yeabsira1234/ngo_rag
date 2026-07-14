from pathlib import Path

import pytest

from src.config import ConfigurationError, Settings


def test_settings_use_current_application_defaults() -> None:
    settings = Settings.from_env({"OPENAI_API_KEY": "test-key"})

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.llm_model == "gpt-4.1-mini"
    assert settings.document_path == Path("data/InternationalHandbook.pdf")
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 150
    assert settings.chroma_collection_name == "ngo_documents"
    assert settings.chroma_persist_directory == Path("chroma_data")
    assert settings.retrieval_result_count == 4


def test_settings_read_environment_overrides() -> None:
    settings = Settings.from_env(
        {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_EMBEDDING_MODEL": "embedding-model",
            "OPENAI_LLM_MODEL": "llm-model",
            "DOCUMENT_PATH": "data/another.pdf",
            "CHUNK_SIZE": "1000",
            "CHUNK_OVERLAP": "200",
            "CHROMA_COLLECTION_NAME": "documents",
            "CHROMA_PERSIST_DIRECTORY": "vector_data",
            "RETRIEVAL_RESULT_COUNT": "6",
        }
    )

    assert settings.openai_api_key == "test-key"
    assert settings.embedding_model == "embedding-model"
    assert settings.llm_model == "llm-model"
    assert settings.document_path == Path("data/another.pdf")
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.chroma_collection_name == "documents"
    assert settings.chroma_persist_directory == Path("vector_data")
    assert settings.retrieval_result_count == 6


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
