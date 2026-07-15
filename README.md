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
- Streamlit chat interface with visible browser-session history
- Versioned FastAPI question-answering endpoint and health check
- Multi-tool OpenAI agent with bounded tool-call iterations
- Safe fictional structured organization-information tool
- API-key redaction in application logs
- Unit tests that do not require OpenAI or a real Chroma database

## Architecture

Ingestion and question answering are separate workflows:

```text
PDF
  -> PDFLoader (one document per page)
  -> TextChunker (metadata-preserving chunks)
  -> OpenAIEmbeddingService
  -> ChromaVectorStore

Question
  -> api.py, streamlit_app.py, or chat.py
  -> RAGService
      -> OpenAIEmbeddingService
      -> ChromaVectorStore (typed results + distance filtering)
      -> RAGPromptBuilder
      -> OpenAILLMService
  -> RAGResponse (answer, status, citations, LLM-called flag)

Agent question
  -> agent_chat.py
  -> AgentService
      -> InMemoryConversationMemory (complete session turns)
      -> OpenAIAgentModel (select a tool or answer directly)
      -> ToolRegistry
          -> DocumentSearchTool -> existing RAGService
          -> OrganizationInfoTool -> fictional structured sample data
  -> AgentResponse (answer, agent status, preserved document citations)
```

`api.py`, `streamlit_app.py`, and `chat.py` are intentionally thin boundary
layers. They use the same application factory and do not contain retrieval,
relevance-filtering, or prompt-building logic.

`AgentService` does not replace `RAGService`. It owns only model-directed tool
selection, safe tool dispatch, and the bounded tool-call loop. The direct CLI,
Streamlit interface, and FastAPI endpoint continue to call `RAGService`
without agent behavior or conversation memory.

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
| `DOCUMENT_PATH` | PDF to ingest | `data/samples/sample_document.pdf` |
| `CHUNK_SIZE` | Chunk size in characters | `800` |
| `CHUNK_OVERLAP` | Character overlap between chunks | `150` |
| `CHROMA_COLLECTION_NAME` | Chroma collection | `ngo_documents` |
| `CHROMA_PERSIST_DIRECTORY` | Local Chroma data directory | `chroma_data` |
| `RETRIEVAL_RESULT_COUNT` | Maximum candidates requested | `4` |
| `RETRIEVAL_MAX_DISTANCE` | Maximum accepted L2 distance | `0.9` |
| `AGENT_MAX_TOOL_ITERATIONS` | Maximum agent tool-call rounds | `2` |
| `AGENT_MEMORY_MAX_TURNS` | Complete agent turns retained in memory | `10` |

`.env` is ignored by Git. Never commit API keys or other credentials.

## Documents and privacy

The repository permits committed PDFs only under `data/samples/`. A document
must be reviewed and confirmed safe before it is added there.

Place internal or sensitive documents under `data/private/`, for example:

```text
data/private/handbook.pdf
```

Then update `.env`:

```env
DOCUMENT_PATH=data/private/handbook.pdf
```

PDFs outside `data/samples/` are ignored by Git. This is a repository safeguard,
not a complete data-governance solution. During ingestion, extracted text is
sent to the configured OpenAI embedding API. Retrieved passages are also sent
to the answer-generation API. Confirm organizational approval and applicable
data-handling requirements before using internal documents.

## Ingesting a PDF

Set `DOCUMENT_PATH`, then run:

```bash
python ingest.py
```

Ingestion loads every PDF page, creates metadata-preserving chunks, requests
embeddings, and writes them to the configured Chroma directory. A successful run
reports the number of stored chunks.

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

The agent currently has two read-only tools and process-local conversation
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
`AGENT_MAX_TOOL_ITERATIONS` prevents an infinite tool loop. The direct RAG CLI,
Streamlit UI, and FastAPI API remain stateless; memory is enabled only for
`agent_chat.py` in this step.

## Running the Streamlit app

Streamlit is installed through `requirements.txt`. After ingestion, start the
web interface from the repository root:

```bash
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit. The interface displays answers and
compact source citations, shows the active document and retrieval configuration
in the sidebar, and provides a button to clear visible chat history.

Streamlit session state preserves messages only so the page can redraw them
during the current browser session. This is not semantic conversation memory:
previous messages are not sent to `RAGService` or the LLM, and each question is
answered independently.

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

## Known limitations

- One configured PDF is ingested per command invocation.
- Chunking is character-based rather than token-, sentence-, or section-aware.
- Image-only PDFs require OCR, which is not currently implemented.
- The relevance threshold has not been evaluated on a labeled benchmark.
- Re-ingestion does not yet remove stale records when chunk identifiers change.
- Visible Streamlit history is not conversational memory.
- The agent has only document-search and fictional organization-information
  tools.
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
2. Approved production tools such as read-only SQL Server queries and approved
   organizational APIs
3. Docker packaging
4. Monitoring, tracing, and operational dashboards
5. CI/CD and security controls
6. Azure deployment and managed production infrastructure

Each phase should preserve the typed RAG service as the core application layer
and add production controls in proportion to deployment risk.
