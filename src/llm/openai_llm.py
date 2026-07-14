from openai import OpenAI

from src.prompting import Prompt


class OpenAILLMService:
    """Generate an answer from a provider-independent prompt using OpenAI."""

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
        prompt: Prompt,
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=prompt.instructions,
            input=prompt.user_input,
        )

        return response.output_text
