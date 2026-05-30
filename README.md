# tech-doc-rag

Viewer-friendly documentation RAG agent that demonstrates the core architecture behind a production agentic retrieval app: SSE streaming, trace visibility, local document embeddings, and session-scoped upload retrieval.

The agent answers questions from bundled LangGraph/tracing documentation, can search the web when current information is useful, can fetch result URLs for supporting evidence, can run restricted Python calculations, and can answer against session-uploaded Markdown, text, PDF, or DOCX files.

## Live Demo

- App: <https://vanmai40-agentic-rag.hf.space/tech-doc.html>
- Health check: <https://vanmai40-agentic-rag.hf.space/health>
- Use the App link above for a quick browser demo with chat, upload, streaming, and TRACE visibility.

This project is hosted live so reviewers can test the actual behavior, not just read code or look at screenshots. That is intentional: many portfolio projects stop at a repository, while this one exposes the running viewer, SSE stream, trace timeline, document retrieval, and upload flow for hands-on evaluation.

The Hugging Face Space is public for portfolio review. Source code is visible, while runtime API keys are stored as Hugging Face Space secrets and are not committed to this repository.

## What This Project Shows

- End-to-end request flow from browser UI to FastAPI to LangGraph tools
- Agent-decided ReAct tool orchestration with LangGraph
- Local document embedding with Hugging Face sentence transformers and FAISS
- Session-scoped upload support that builds a temporary retrieval index per upload
- Server-sent event streaming for token output and tool progress
- TRACE event building so viewers can inspect routing, retrieval, web search, uploads, and final-answer steps
- DuckDuckGo/DDG web search without a paid search API key
- URL fetching for secondary evidence from search results
- Restricted Python execution for calculations
- Public-demo hardening for upload size, file type, trace redaction, and error handling
- Hugging Face Spaces deployment from the repo-root Dockerfile

## Architecture

```text
Browser viewer
  - chat composer
  - upload control
  - streaming answer area
  - TRACE timeline
        |
        | HTTP JSON + SSE stream
        v
FastAPI app
  - serves the static viewer
  - validates uploads
  - emits trace events
        |
        v
LangGraph ReAct loop
  - agent node chooses the next action
  - tool node executes retrieval/search/fetch/python
        |
        +-- retrieve_docs -> bundled FAISS index + uploaded-session FAISS index
        +-- search_web    -> DDG web results
        +-- fetch_url      -> secondary URL fetch from search results
        +-- run_python     -> restricted Python calculations
        v
OpenAI-compatible LLM endpoint
```

The graph loops between an LLM-powered agent node and a tool executor until the model returns a final answer. The SSE endpoint streams both answer chunks and structured TRACE events, so a reviewer can see what the agent is doing instead of treating the response as a black box. The same backend serves the browser UI, non-streaming chat, streaming chat, document listing, upload, and reset endpoints.

## Current Stack

| Layer | Current choice |
|---|---|
| Agent orchestration | LangGraph ReAct loop |
| API | FastAPI + Uvicorn |
| Frontend | Static HTML served by FastAPI |
| LLM | OpenAI-compatible chat endpoint |
| Embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` by default |
| Vector store | Local FAISS index |
| Web search | DuckDuckGo/DDG HTML search |
| Tracing | In-app TRACE panel and SSE trace events |
| Deployment | Hugging Face Spaces Docker app |

Pinecone, Tavily, and LangSmith accounts are not required for the current implementation.

## Key Files

```text
tech-doc-rag/
├── src/
│   ├── graph.py       # LangGraph ReAct loop and routing
│   ├── tools.py       # retrieval, web search, URL fetch, Python, uploads
│   ├── prompts.py     # ReAct system prompt and answer rules
│   ├── ingest.py      # builds the local FAISS index
│   ├── config.py      # Pydantic settings from the repo-root .env
│   └── main.py        # terminal chat entry point
├── data/              # bundled markdown documentation corpus
├── tests/             # graph, upload, endpoint, and regression tests
├── .faiss_index/      # local vector index used by the deployed app
└── README.md
```

The production container is built from the repository root, not this subdirectory. The root `Dockerfile` copies `api/`, `frontend/`, `tech-doc-rag/src/`, `tech-doc-rag/data/`, and `tech-doc-rag/.faiss_index/` into the image.

## Quick Start

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt
cp .env.example .env
```

For local smoke tests without an LLM key, keep `MOCK_LLM=true` in `.env`. For real answers, set:

```bash
MOCK_LLM=false
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3
EMBEDDING_PROVIDER=huggingface
USE_LOCAL_VECTORSTORE=true
```

The repository already includes a local FAISS index for the bundled documentation. If you change the bundled docs and want to rebuild the index, run the ingest module from `tech-doc-rag/` with the project dependencies installed, including the optional Pinecone packages used by the ingest script:

```bash
cd tech-doc-rag
python -m src.ingest
cd ..
```

Start the local API and browser UI:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/tech-doc.html?nocache=1
```

## API Surface

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `GET /tech-doc.html` | Browser chat UI |
| `GET /api/tech-doc/documents` | List bundled docs |
| `POST /api/tech-doc/chat` | Non-streaming chat response |
| `POST /api/tech-doc/chat/stream` | SSE streaming chat with TRACE events |
| `POST /api/tech-doc/documents/upload` | Upload a session document |
| `DELETE /api/tech-doc/documents/upload/{session_id}` | Reset uploaded document by session |
| `POST /api/tech-doc/documents/upload/reset` | Reset uploaded document by request body |

Uploads accept `.md`, `.markdown`, `.txt`, `.pdf`, and `.docx` files up to 2 MB.

## Example Requests

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/api/tech-doc/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"How does StateGraph work in LangGraph?"}'

curl -N -X POST http://localhost:8000/api/tech-doc/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"What is 2+2?"}'
```

## Sample Questions

```text
How does StateGraph work in LangGraph?
How can the TRACE panel help evaluate a workflow?
What changed recently in LangGraph?
Upload this PDF and summarize the deployment requirements.
What is 17 * 23?
```

## Testing

Run the focused tests that exercise the published FastAPI surface and upload/RAG behavior:

```bash
pytest tech-doc-rag/tests/test_api_integration.py tech-doc-rag/tests/test_upload_rag.py -q
```

Or run the local smoke script from the repository root while the API is running:

```bash
python test_local.py
```

The broader historical test suite may include development-time regression coverage from the original multi-project workspace. For this standalone package, the smoke script and focused API/upload tests are the most relevant checks.

## Deployment

The current production target is Hugging Face Spaces with Docker SDK, public visibility, repository-root build context, and exposed port `7860`.

Required production values:

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`

Do not commit real `.env` files or API keys. Use Hugging Face Space secrets for production credentials.

The live demo currently runs on Hugging Face Spaces at <https://vanmai40-agentic-rag.hf.space/tech-doc.html>.
