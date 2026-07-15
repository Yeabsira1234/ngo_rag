# Document RAG Assistant

A production-minded retrieval-augmented generation (RAG) foundation for
answering questions from PDF documents. The project extracts page-aware text,
creates OpenAI embeddings, stores them in ChromaDB, retrieves relevant chunks,
and generates grounded answers with citations.

The project includes a basic tool-calling agent, FastAPI HTTP API, Streamlit
web chat, and direct RAG command-line interface. The agent is a separate layer
that can choose document retrieval, fictional structured organization data,
or a direct answer.

## Current features

- Page-aware PDF loading with PyMuPDF
- Overlapping text chunking with preserved metadata
- OpenAI embeddings and answer generation
- Persistent local ChromaDB vector storage
- Configurable L2 relevance filtering
- Source, PDF page, chunk, and distance citations
- Typed retrieval and RAG response models
- Centralized prompt construction and RAG orchestration
- Safe CLI error handling and structured file logging
- Streamlit agent chat with isolated in-session conversation memory
- Versioned FastAPI question-answering endpoint and health check
- Multi-tool OpenAI agent with bounded tool-call iterations
- Safe fictional structured organization-information tool
- API-key redaction in application logs
- Unit tests that do not require OpenAI or a real Chroma database

## Architecture

Ingestion and question answering are separate workflows:

```text
Configured PDF directory
  -> PDFDocumentDiscovery (stable, validated collection)
  -> PDFLoader (one document per page)
  -> TextChunker (metadata-preserving chunks)
  -> OpenAIEmbeddingService
  -> ChromaVectorStore

Direct RAG question
  -> api.py or chat.py
  -> RAGService
      -> OpenAIEmbeddingService
      -> ChromaVectorStore (typed results + distance filtering)
      -> RAGPromptBuilder
      -> OpenAILLMService
  -> RAGResponse (answer, status, citations, LLM-called flag)

Agent question
  -> agent_chat.py or streamlit_app.py
  -> AgentService
      -> InMemoryConversationMemory (complete session turns)
      -> OpenAIAgentModel (select a tool or answer directly)
      -> ToolRegistry
          -> DocumentSearchTool -> existing RAGService
          -> OrganizationInfoTool -> fictional structured sample data
          -> SQLQueryTool -> predefined read-only SQL operations
  -> AgentResponse (answer, agent status, preserved document citations)
```

`api.py`, `streamlit_app.py`, and `chat.py` are intentionally thin boundary
layers. They use application factories and do not contain retrieval,
relevance-filtering, prompt-building, tool-routing, or memory logic.

`AgentService` does not replace `RAGService`. It owns model-directed tool
selection, safe tool dispatch, bounded tool calls, and conversation memory.
Streamlit and `agent_chat.py` use `AgentService`; the direct `chat.py` CLI and
FastAPI endpoint continue to use `RAGService` without agent behavior.

## Repository structure

```text
.
|-- chat.py                         # Interactive chat CLI
|-- agent_chat.py                   # Tool-calling agent CLI
|-- api.py                          # FastAPI HTTP entry point
|-- streamlit_app.py               # Streamlit web chat
|-- ingest.py                       # PDF ingestion CLI
|-- requirements.txt               # Direct Python dependencies
|-- .env.example                   # Safe configuration template
|-- data/
|   |-- samples/                    # Explicitly reviewed, safe sample PDFs
|   `-- private/                    # Internal PDFs; ignored by Git
|-- src/
|   |-- chunking/text_chunker.py    # Metadata-preserving chunking
|   |-- embeddings/openai_embeddings.py
|   |-- llm/openai_llm.py
|   |-- loaders/pdf_loader.py
|   |-- vectorstore/chroma_store.py
|   |-- config.py                   # Typed environment settings
|   |-- application.py              # Shared dependency factory
|   |-- agent/                       # Agent models, tools, and orchestration
|   |   `-- memory.py                # Typed process-local conversation memory
|   |   `-- organization_data.py     # Fictional structured sample information
|   |-- api_models.py               # Typed HTTP request/response schemas
|   |-- chat_history.py             # Visible UI-session message models
|   |-- documents.py                # Page and chunk document models
|   |-- logging_config.py           # Centralized logging and redaction
|   |-- prompting.py                # Provider-independent RAG prompts
|   |-- rag_service.py              # Question-answering orchestration
|   `-- retrieval.py                # Typed retrieval results
`-- tests/                          # Unit and CLI boundary tests
```

## Prerequisites

- Python 3.11 or newer
- An OpenAI API key
- Internet access for ingestion and answer generation

ChromaDB runs locally and does not require a separate database server.

## Installation

Clone the repository and enter its directory, then create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the direct dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment configuration

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

At minimum, replace the placeholder API key:

```env
OPENAI_API_KEY=your-api-key
```

The available settings are:

| Setting | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI authentication; required | none |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `OPENAI_LLM_MODEL` | Answer-generation model | `gpt-4.1-mini` |
| `LOG_LEVEL` | Application log verbosity | `INFO` |
| `DOCUMENTS_DIRECTORY` | Directory containing PDFs to ingest | `data/samples` |
| `DOCUMENT_GLOB` | PDF discovery pattern | `*.pdf` |
| `CHUNK_SIZE` | Chunk size in characters | `800` |
| `CHUNK_OVERLAP` | Character overlap between chunks | `150` |
| `CHROMA_COLLECTION_NAME` | Chroma collection | `ngo_documents` |
| `CHROMA_PERSIST_DIRECTORY` | Local Chroma data directory | `chroma_data` |
| `RETRIEVAL_RESULT_COUNT` | Maximum candidates requested | `4` |
| `RETRIEVAL_MAX_DISTANCE` | Maximum accepted L2 distance | `0.9` |
| `AGENT_MAX_TOOL_ITERATIONS` | Maximum agent tool-call rounds | `2` |
| `AGENT_MEMORY_MAX_TURNS` | Complete agent turns retained in memory | `10` |

The old `DOCUMENT_PATH` setting was removed rather than deprecated: ingestion
now always targets one explicitly configured directory and glob, avoiding two
competing sources of truth.

`.env` is ignored by Git. Never commit API keys or other credentials.

## Documents and privacy

The repository permits committed PDFs only under `data/samples/`. A document
must be reviewed and confirmed safe before it is added there.

Place internal or sensitive PDFs under `data/private/`. To include nested
folders, use `**/*.pdf`:

```text
data/private/handbook.pdf
data/private/policies/leave.pdf
```

Then update `.env`:

```env
DOCUMENTS_DIRECTORY=data/private
DOCUMENT_GLOB=**/*.pdf
```

PDFs outside `data/samples/` are ignored by Git. This is a repository safeguard,
not a complete data-governance solution. During ingestion, extracted text is
sent to the configured OpenAI embedding API. Retrieved passages are also sent
to the answer-generation API. Confirm organizational approval and applicable
data-handling requirements before using internal documents.

## Ingesting a document collection

Set `DOCUMENTS_DIRECTORY` and `DOCUMENT_GLOB`, then run:

```bash
python ingest.py
```

Discovery sorts matching PDFs deterministically and ignores hidden, temporary,
zero-byte, directory, and unsupported entries. Ingestion continues when an
individual PDF is unreadable, reports failures, and succeeds only if at least
one document produces chunks. It reports discovered and processed documents,
pages, chunks, and skipped or failed documents.

Each page and chunk carries the filename, collection-relative path, physical
page number, per-document chunk index, and a stable `document_id`. The document
ID is the first 24 hexadecimal characters of SHA-256 over the normalized
relative path; it never exposes an absolute path and distinguishes equal
filenames in different folders. Chunk indices restart at zero for each document.
Chroma IDs combine `document_id`, page number, and chunk index.

Ingestion uses deterministic Chroma upserts, so running it repeatedly does not
duplicate unchanged chunks. Collection-wide stale deletion is disabled because
CLI and browser ingestion manage different directories and must not delete one
another's vectors. Changing a file's contents without changing its path reuses
record IDs; if the new version produces fewer chunks, surplus old chunks for
that same document require rebuilding the local Chroma directory.

If the stored schema, chunking configuration, embedding model, or source
document changes, rebuild the generated Chroma data before evaluating retrieval.
Chroma data is generated locally and ignored by Git.

## Running the chat CLI

After ingestion:

```bash
python chat.py
```

Enter a document-related question. Use `exit` or `quit` to stop the application.
The CLI calls `RAGService`; `rag_service.py` is not a standalone executable.

## Running the agent CLI

After ingestion, start the separate tool-calling agent:

```bash
python agent_chat.py
```

Unlike `chat.py`, which always performs document retrieval, the agent asks the
configured OpenAI model to choose among:

- `document_search` for policies, procedures, and facts contained in the
  indexed PDF. This uses the existing `RAGService` and preserves citations.
- `organization_info` for one structured fact from a small fictional sample
  directory. Results identify `organization_info` as their source and do not
  have document citations.
- A direct answer when neither tool is required.

Example agent questions:

```text
What policy does the indexed document give for remote-work requests?
What is the fictional organization's general contact email?
Where is its main office?
What about its service categories?
Explain what retrieval-augmented generation means.
```

The organization data is deliberately fictional and stored separately from
the indexed PDF. It is not USCRI, university, personal, or private
organizational information.

The agent currently has three read-only tools and process-local conversation
memory. Follow-up questions receive retained user/assistant turns and any
ordered tool-call context required by the Responses API. Enter `/clear` to
remove the current session history without exiting.

Memory retains at most `AGENT_MEMORY_MAX_TURNS` complete exchanges. When the
limit is exceeded, the oldest whole turn is removed, including its tool call
and matching result. History is never written to disk and disappears when the
agent process exits. Each new `agent_chat.py` process receives an independent
memory store. A future deployment can replace the small memory-store interface
with Redis or a database without moving state into `RAGService`.

Unknown tools and malformed arguments are rejected, and
`AGENT_MAX_TOOL_ITERATIONS` prevents an infinite tool loop. The direct RAG CLI
and FastAPI API remain stateless. Memory is enabled for `agent_chat.py` and
independently for each Streamlit browser session.

## Running the Streamlit app

Streamlit is installed through `requirements.txt`. After ingestion, start the
web interface from the repository root:

```bash
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit. The web chat uses `AgentService`, so it
supports document search, fictional organization information, predefined
read-only SQL queries, direct answers, and contextual follow-up questions.
Assistant messages display compact status, safe tool labels, and document
citations without exposing tool payloads, SQL, or provider internals.

Each browser session owns a separate agent and in-memory conversation store in
`st.session_state`. Reruns preserve that session's context, while different
browser sessions do not share memory. **Clear chat history** clears both visible
messages and underlying agent memory. Memory remains non-persistent and is lost
when the browser session or application process ends.

### Browser PDF ingestion

The document-management section accepts up to `MAX_UPLOAD_FILES` PDF files per
batch, with each file limited to `MAX_UPLOAD_FILE_SIZE_MB`. Selection alone does
not ingest anything; press **Upload and ingest documents** explicitly. Files are
validated as readable PDFs before any member of the batch is saved.

Uploads are stored under `UPLOAD_DIRECTORY` (`data/uploads` by default), which
is ignored by Git. Filenames must be simple visible PDF names without path
components or suspicious characters. Temporary files are atomically moved into
place only after the whole batch passes validation and are cleaned afterward.
The UI never displays extracted text, embeddings, credentials, stack traces, or
absolute local paths.

Existing filenames are never silently replaced. Re-uploading byte-identical
content is treated as unchanged and skips ingestion; different content under an
existing filename is rejected. Browser uploads use a separate identity namespace
so a sample/private PDF with the same relative filename cannot collide. Upload
ingestion uses the same discovery, loading, chunking, embedding, and Chroma
services as `ingest.py`, and successful documents are immediately available to
chat. A process-local lock prevents overlapping browser ingestion runs.

```env
UPLOAD_DIRECTORY=data/uploads
MAX_UPLOAD_FILE_SIZE_MB=10
MAX_UPLOAD_FILES=10
```

Because CLI and browser ingestion can manage different directories, Step 15
does not perform collection-wide stale-vector deletion during ingestion; doing
so could delete unrelated vectors from another source. Browser replacement is
rejected, so it cannot leave stale chunks for an existing upload. Production
storage, coordinated cleanup, authentication, and asynchronous ingestion are
deferred to the later LLMOps phase.

Visible Streamlit messages remain separate from internal agent context and hold
only redraw data. Upload ingestion updates the same Chroma collection used by
the session agent, so new documents are searchable immediately without
recreating conversation memory.

## Running the FastAPI service

FastAPI and Uvicorn are installed through `requirements.txt`. After configuring
the environment and ingesting a document, start the development API from the
repository root:

```bash
python -m uvicorn api:app --reload
```

The service is available at `http://127.0.0.1:8000`. Interactive Swagger UI is
available at:

```text
http://127.0.0.1:8000/docs
```

The health endpoint does not call OpenAI:

```bash
curl http://127.0.0.1:8000/health
```

Submit a question with:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the support office hours?"}'
```

An answered response has this shape:

```json
{
  "answer": "The support office is open Monday through Friday...",
  "status": "answered",
  "llm_called": true,
  "citations": [
    {
      "source": "sample_document.pdf",
      "page_number": 1,
      "chunk_index": 0,
      "distance": 0.7207
    }
  ]
}
```

API status behavior:

- `200`: answered or insufficient-context result. Insufficient context is an
  expected RAG outcome, not an infrastructure error.
- `422`: missing, malformed, empty, or whitespace-only question.
- `503`: embedding, retrieval, Chroma, or LLM dependency failure.
- `500`: unexpected application failure.

Error responses contain safe generic messages; technical details are written
to the application log. Authentication and authorization are not implemented
yet, so this API must not be exposed publicly.

## Relevance filtering

The current Chroma collection uses L2 distance:

- Lower distance means a closer embedding match.
- `0.0` represents identical embeddings.
- A result is accepted when its distance is less than or equal to
  `RETRIEVAL_MAX_DISTANCE`.

The default threshold of `0.9` is an initial corpus-specific baseline, not a
universal optimum. If no retrieved chunks pass the threshold, the LLM is not
called and the assistant returns an insufficient-context response. Thresholds
should eventually be calibrated using a labeled retrieval evaluation dataset.

## Citations

Every stored chunk retains:

- Source filename
- One-based physical PDF page number
- Document-wide, zero-based chunk index
- Retrieval distance

The physical PDF page can differ from a page number printed inside the document
when a cover or front matter is present. A CLI citation looks like:

```text
sample_document.pdf, page 7, chunk 31, distance 0.7321
```

## Logging

Structured application logs are written to:

```text
logs/application.log
```

The log level is controlled by `LOG_LEVEL`. Logs include lifecycle events,
counts, failure types, and stack traces. They intentionally exclude full user
questions, extracted document contents, prompts, and API keys. The `logs/`
directory is ignored by Git.

## Running tests

Run the complete test suite from the repository root:

```bash
python -m pytest -q
```

The unit tests use mocks or temporary local resources. They do not make OpenAI
requests or require the project's persistent Chroma database.

## Read-only SQL Server agent tool

The agent can answer structured questions from the local `NGO_RAG` SQL Server
database through the `sql_query` tool. Install Microsoft ODBC Driver 18 for SQL
Server on the machine before using it. Local development uses Windows
Authentication; no database password is stored in source code.

Configure `.env` locally (the application never writes this file):

```env
SQL_SERVER=YEABSIRA
SQL_DATABASE=NGO_RAG
SQL_DRIVER=ODBC Driver 18 for SQL Server
SQL_TRUSTED_CONNECTION=true
SQL_TRUST_SERVER_CERTIFICATE=true
SQL_QUERY_TIMEOUT_SECONDS=10
SQL_MAX_ROWS=100
```

Supported operations are `list_offices`, `list_programs`,
`count_cases_by_status`, `list_programs_by_office`, `list_staff_by_office`,
`list_open_cases`, `count_clients_by_language`, and
`recent_service_events`. Each operation maps to fixed read-only SQL with
explicit columns, parameterized values, deterministic ordering, a timeout, and
a maximum result size. Arbitrary SQL, stored procedures, and write operations
are not accepted.

Example agent questions include “How many open cases are there?”, “Which
programs are offered by the Northern Virginia Office?”, “How many clients
prefer Amharic?”, and “What recent services were provided?”. Unrestricted
natural-language-to-SQL is intentionally deferred to a later step.

## Known limitations

- Browser uploads use process-local storage and concurrency control; they are
  not suitable for horizontally scaled deployment yet.
- Chunking is character-based rather than token-, sentence-, or section-aware.
- Image-only PDFs require OCR, which is not currently implemented.
- The relevance threshold has not been evaluated on a labeled benchmark.
- Re-ingestion does not yet remove stale records when chunk identifiers change.
- Streamlit agent memory is isolated per browser session but is not durable.
- The agent has document-search, fictional organization-information, and
  predefined read-only SQL tools only.
- Structured organization data is static sample data, not a production system
  of record.
- Agent memory is not shared across processes and is not durable.
- Agent tool selection does not yet have a labeled evaluation benchmark.
- There is no authentication or user authorization.
- OpenAI and local Chroma failures are logged but are not retried.
- There is no automated RAG evaluation or production monitoring yet.

## Roadmap

Planned work will be introduced incrementally:

1. RAG evaluation datasets, retrieval metrics, and answer-quality evaluation
2. Approved production tools such as additional organizational APIs and
   carefully governed natural-language-to-SQL
3. Docker packaging
4. Monitoring, tracing, and operational dashboards
5. CI/CD and security controls
6. Azure deployment and managed production infrastructure

Each phase should preserve the typed RAG service as the core application layer
and add production controls in proportion to deployment risk.
