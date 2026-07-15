import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from src.api_models import AskRequest, AskResponse, HealthResponse
from src.application import build_rag_service
from src.config import Settings
from src.logging_config import configure_logging
from src.rag_service import RAGDependencyError, RAGService


logger = logging.getLogger(__name__)
SERVICE_UNAVAILABLE_MESSAGE = "The document assistant is temporarily unavailable."
INTERNAL_ERROR_MESSAGE = "The document assistant could not process the request."


def create_app(rag_service: RAGService | None = None) -> FastAPI:
    """Create the API, optionally using an injected service for testing."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if rag_service is not None:
            application.state.rag_service = rag_service
            yield
            return

        settings: Settings | None = None
        try:
            settings = Settings.from_env()
            configure_logging(
                settings.log_level,
                sensitive_values=(settings.openai_api_key,),
            )
            logger.info("event=api_startup_started")
            application.state.rag_service = build_rag_service(settings)
        except Exception as error:
            sensitive_values = (
                (settings.openai_api_key,) if settings is not None else ()
            )
            configure_logging("ERROR", sensitive_values=sensitive_values)
            logger.exception(
                "event=api_startup_failed error_type=%s",
                type(error).__name__,
            )
            raise

        logger.info("event=api_startup_completed")
        yield
        logger.info("event=api_shutdown")

    application = FastAPI(
        title="Document RAG Assistant API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @application.post("/api/v1/ask", response_model=AskResponse)
    def ask(payload: AskRequest, request: Request) -> AskResponse:
        service: RAGService = request.app.state.rag_service
        try:
            response = service.answer(payload.question)
        except RAGDependencyError as error:
            logger.error(
                "event=api_request_dependency_failed error_type=%s "
                "question_length=%d",
                type(error).__name__,
                len(payload.question),
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=SERVICE_UNAVAILABLE_MESSAGE,
            ) from error
        except Exception as error:
            logger.exception(
                "event=api_request_unexpected_failure error_type=%s "
                "question_length=%d",
                type(error).__name__,
                len(payload.question),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=INTERNAL_ERROR_MESSAGE,
            ) from error

        return AskResponse.from_rag_response(response)

    return application


app = create_app()
