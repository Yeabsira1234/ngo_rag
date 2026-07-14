from dataclasses import dataclass
from typing import Sequence

from src.retrieval import RetrievalResult


@dataclass(frozen=True, slots=True)
class Prompt:
    """Provider-independent prompt content for an LLM request."""

    instructions: str
    user_input: str

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise ValueError("instructions cannot be empty.")
        if not self.user_input.strip():
            raise ValueError("user_input cannot be empty.")


class RAGPromptBuilder:
    """Build a grounded question-answering prompt from retrieved chunks."""

    def build(
        self,
        question: str,
        results: Sequence[RetrievalResult],
    ) -> Prompt:
        if not question.strip():
            raise ValueError("question cannot be empty.")
        if not results:
            raise ValueError("results cannot be empty.")

        context = "\n\n---\n\n".join(
            (
                f"[Source: {result.source}, page {result.page_number}, "
                f"chunk {result.chunk_index}]\n{result.chunk_text}"
            )
            for result in results
        )

        instructions = (
            "You are a document question-answering assistant. "
            "Answer only from the supplied context. "
            "If the answer is not contained in the context, say that you "
            "could not find the answer in the document. "
            "Do not invent names, phone numbers, policies, or facts."
        )
        user_input = f"Context:\n{context}\n\nQuestion:\n{question}"

        return Prompt(
            instructions=instructions,
            user_input=user_input,
        )
