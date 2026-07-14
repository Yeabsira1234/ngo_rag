import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A retrieved chunk and the information needed to cite and rank it."""

    chunk_text: str
    source: str
    page_number: int
    chunk_index: int
    distance: float

    def __post_init__(self) -> None:
        if not self.chunk_text.strip():
            raise ValueError("chunk_text cannot be empty.")
        if not self.source.strip():
            raise ValueError("source cannot be empty.")
        if self.page_number <= 0:
            raise ValueError("page_number must be greater than zero.")
        if self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative.")
        if not math.isfinite(self.distance) or self.distance < 0:
            raise ValueError("distance must be a finite, non-negative value.")

    @classmethod
    def from_chroma(
        cls,
        chunk_text: str,
        metadata: Mapping[str, object],
        distance: float,
    ) -> "RetrievalResult":
        """Map one Chroma result into the application's typed model."""
        source = metadata.get("source")
        if not isinstance(source, str):
            raise ValueError("Chroma metadata must contain a string source.")

        return cls(
            chunk_text=chunk_text,
            source=source,
            page_number=_required_int(metadata, "page_number"),
            chunk_index=_required_int(metadata, "chunk_index"),
            distance=float(distance),
        )

    def is_relevant(self, max_distance: float) -> bool:
        """Return whether this result is within the accepted L2 distance."""
        if not math.isfinite(max_distance) or max_distance < 0:
            raise ValueError(
                "max_distance must be a finite, non-negative value."
            )
        return self.distance <= max_distance


def _required_int(metadata: Mapping[str, object], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Chroma metadata must contain an integer {key}.")
    return value
