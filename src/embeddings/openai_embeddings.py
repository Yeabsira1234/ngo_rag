from openai import OpenAI


class OpenAIEmbeddingService:
    def __init__(
        self,
        api_key: str,
        model: str,
    ) -> None:
        if not api_key:
            raise ValueError("api_key cannot be empty.")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        if not query.strip():
            raise ValueError("The query cannot be empty.")

        response = self.client.embeddings.create(
            model=self.model,
            input=query,
        )

        return response.data[0].embedding
