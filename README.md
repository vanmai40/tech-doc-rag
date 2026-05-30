# tech-doc-rag

Viewer-friendly documentation RAG agent that demonstrates the core architecture behind a production agentic retrieval app: SSE streaming, trace visibility, local document embeddings, and session-scoped upload retrieval.

The agent answers questions from bundled LangGraph/tracing documentation, can search the web when current information is useful, can fetch result URLs for supporting evidence, can run restricted Python calculations, and can answer against session-uploaded Markdown, text, PDF, or DOCX files.

## Live Demo

- App: <https://vanmai40-agentic-rag.hf.space/tech-doc.html>
- Health check: <https://vanmai40-agentic-rag.hf.space/health>
- Use the App link above for a quick browser demo with chat, upload, streaming, and TRACE visibility.

This project is hosted live so reviewers can test the actual behavior, not just read code or look at screenshots. That is intentional: many portfolio projects stop at a repository, while this one exposes the running viewer, SSE stream, trace timeline, document retrieval, and upload flow for hands-on evaluation.

The Hugging Face Space is public for portfolio review. Source code is visible, while runtime API keys are stored as Hugging Face Space secrets and are not committed to this repository.

![Live tech-doc-rag chat demo](assets/chat-demo.png)

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

## Try These In The Live Demo

```text
How does StateGraph work in LangGraph?
How can the TRACE panel help evaluate a workflow?
What changed recently in LangGraph?
Upload this PDF and summarize the deployment requirements.
What is 17 * 23?
```
