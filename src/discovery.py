import hashlib
from dataclasses import dataclass
from pathlib import Path


class DocumentDiscoveryError(RuntimeError):
    """Raised when a configured collection contains no valid documents."""


@dataclass(frozen=True, slots=True)
class DiscoveredDocument:
    path: Path
    relative_path: str
    document_id: str


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    documents: tuple[DiscoveredDocument, ...]
    skipped_count: int


class PDFDocumentDiscovery:
    def discover(
        self,
        directory: str | Path,
        pattern: str = "*.pdf",
        *,
        identity_namespace: str = "",
    ) -> DiscoveryResult:
        root = Path(directory)
        if not root.is_dir():
            raise DocumentDiscoveryError("The configured documents directory is unavailable.")
        candidates = sorted(root.glob(pattern), key=lambda path: path.as_posix().casefold())
        documents: list[DiscoveredDocument] = []
        seen: set[Path] = set()
        skipped = 0
        for path in candidates:
            resolved = path.resolve()
            relative = path.relative_to(root)
            if (
                resolved in seen
                or not path.is_file()
                or path.suffix.casefold() != ".pdf"
                or any(part.startswith(".") for part in relative.parts)
                or path.name.startswith(("~", "$"))
                or path.stat().st_size == 0
            ):
                skipped += 1
                continue
            seen.add(resolved)
            relative_path = relative.as_posix()
            identity_source = f"{identity_namespace}/{relative_path}" if identity_namespace else relative_path
            identity = hashlib.sha256(identity_source.casefold().encode("utf-8")).hexdigest()[:24]
            documents.append(DiscoveredDocument(path, relative_path, identity))
        if not documents:
            raise DocumentDiscoveryError("No valid PDF documents were found.")
        return DiscoveryResult(tuple(documents), skipped)
