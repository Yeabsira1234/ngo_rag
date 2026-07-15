import hashlib
import logging
from typing import Protocol

from src.sql.generation import GeneratedSQL, SQLGenerationError
from src.sql.models import SQLQueryResult
from src.sql.validation import SQLPrivacyError, SQLValidationError, SQLValidator

logger = logging.getLogger(__name__)


class SQLGenerator(Protocol):
    def generate(self, question: str) -> GeneratedSQL: ...


class GeneratedSQLRepository(Protocol):
    def execute_generated(self, sql: str, parameters: tuple[object, ...]) -> SQLQueryResult: ...


class SQLQuestionPolicy:
    """Reject plainly destructive or broad client-record requests pre-generation."""

    def validate(self, question: str) -> None:
        normalized = " ".join(question.casefold().split())
        forbidden = (
            "delete ", "update ", "insert ", "drop ", "truncate ",
            "xp_", "sp_", "all client", "every client", "client identifiers",
            "all rows", "every row", "reveal schema prompt", "bypass safety",
        )
        if any(fragment in normalized for fragment in forbidden):
            raise SQLPrivacyError("The request violates the read-only data policy.")


class NaturalLanguageSQLService:
    def __init__(self, generator: SQLGenerator, validator: SQLValidator,
                 repository: GeneratedSQLRepository,
                 policy: SQLQuestionPolicy | None = None) -> None:
        self.generator = generator
        self.validator = validator
        self.repository = repository
        self.policy = policy or SQLQuestionPolicy()

    def query(self, question: str) -> SQLQueryResult:
        try:
            self.policy.validate(question)
            candidate = self.generator.generate(question)
            approved = self.validator.validate(candidate)
        except (SQLValidationError, SQLGenerationError) as error:
            logger.warning(
                "event=generated_sql_rejected reason_type=%s",
                type(error).__name__,
            )
            raise
        fingerprint = hashlib.sha256(approved.sql.encode("utf-8")).hexdigest()[:16]
        logger.info("event=generated_sql_approved query_fingerprint=%s", fingerprint)
        return self.repository.execute_generated(approved.sql, approved.parameters)
