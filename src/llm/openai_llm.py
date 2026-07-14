from openai import OpenAI


class OpenAILLMService:
    def __init__(
        self,
        api_key: str,
        model: str,
    ) -> None:
        if not api_key:
            raise ValueError("api_key cannot be empty.")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_answer(
        self,
        question: str,
        context_chunks: list[str],
    ) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not context_chunks:
            return "I could not find relevant information in the document."

        context = "\n\n---\n\n".join(context_chunks)

        instructions = (
            "You are a document question-answering assistant. "
            "Answer only from the supplied context. "
            "If the answer is not contained in the context, say that you "
            "could not find the answer in the document. "
            "Do not invent names, phone numbers, policies, or facts."
        )

        user_input = f"""
Context:
{context}

Question:
{question}
"""

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
        )

        return response.output_text
