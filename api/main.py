from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import json

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tech-doc-rag" / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_static_html(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") or path.endswith(".html") or path == "/":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

_frontend_dir = Path(__file__).parent.parent / "frontend"
if (_frontend_dir / "index.html").exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


@app.get("/")
async def root():
    index_path = _frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    tech_doc_path = _frontend_dir / "tech-doc.html"
    if tech_doc_path.exists():
        return FileResponse(str(tech_doc_path))
    return {"message": "tech-doc-rag API", "docs": "/api/tech-doc/chat"}


@app.get("/tech-doc.html")
async def tech_doc_page():
    page_path = _frontend_dir / "tech-doc.html"
    if page_path.exists():
        return FileResponse(str(page_path))
    return {"error": "Page not found"}


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = Field(default_factory=list)
    upload_session_id: Optional[str] = None
    uploadSessionId: Optional[str] = None

    def effective_upload_session_id(self) -> Optional[str]:
        return self.upload_session_id or self.uploadSessionId


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []


class UploadResetRequest(BaseModel):
    session_id: Optional[str] = None
    upload_session_id: Optional[str] = None
    uploadSessionId: Optional[str] = None

    def effective_session_id(self) -> Optional[str]:
        return self.session_id or self.upload_session_id or self.uploadSessionId


_tech_doc_data_dir = Path(__file__).parent.parent / "tech-doc-rag" / "data"


def _document_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def _load_tech_doc_documents(data_dir: Path = _tech_doc_data_dir) -> list[dict]:
    base_dir = data_dir.resolve()
    if not base_dir.exists():
        return []

    documents = []
    for file_path in sorted(base_dir.rglob("*.md")):
        resolved_path = file_path.resolve()
        try:
            relative_path = resolved_path.relative_to(base_dir)
        except ValueError:
            continue
        if resolved_path.suffix.lower() != ".md" or not resolved_path.is_file():
            continue

        content = resolved_path.read_text(encoding="utf-8")
        stable_path = relative_path.as_posix()
        documents.append({
            "id": stable_path,
            "title": _document_title(content, resolved_path.stem),
            "source": relative_path.parts[0] if len(relative_path.parts) > 1 else resolved_path.stem,
            "path": stable_path,
            "content": content,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": len(content.splitlines()),
        })

    return documents


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/tech-doc/documents")
async def tech_doc_documents():
    return {"documents": _load_tech_doc_documents()}


def _normalize_upload_session_id(value: object) -> str:
    session_id = str(value or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="upload session id is required")
    if len(session_id) > settings.max_upload_session_id_length:
        raise HTTPException(status_code=400, detail="upload session id is too long")
    return session_id


def _validate_upload_filename(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".md", ".markdown", ".txt", ".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="unsupported upload format; upload Markdown, text, PDF, or DOCX")


@app.post("/api/tech-doc/documents/upload")
async def tech_doc_upload_document(request: Request):
    try:
        form = await request.form()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="multipart form data is required") from exc

    session_id = _normalize_upload_session_id(
        form.get("session_id") or form.get("upload_session_id") or form.get("uploadSessionId")
    )
    uploaded_file = form.get("file") or form.get("document")
    if uploaded_file is None or not hasattr(uploaded_file, "read"):
        raise HTTPException(status_code=400, detail="multipart file field is required")

    filename = Path(str(getattr(uploaded_file, "filename", "") or "uploaded-document")).name
    content_type = str(getattr(uploaded_file, "content_type", "") or "")
    _validate_upload_filename(filename)
    data = await uploaded_file.read()
    if len(data) > settings.max_upload_bytes:
        size_mb = settings.max_upload_bytes / 1_000_000
        raise HTTPException(status_code=413, detail=f"uploaded document is too large; limit is {size_mb:.1f} MB")

    from tools import ingest_uploaded_document

    try:
        document = ingest_uploaded_document(session_id, filename, content_type, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"document": document}


@app.delete("/api/tech-doc/documents/upload/{session_id}")
async def tech_doc_reset_upload_document(session_id: str):
    from tools import reset_uploaded_document

    reset_uploaded_document(_normalize_upload_session_id(session_id))
    return {"ok": True}


@app.post("/api/tech-doc/documents/upload/reset")
async def tech_doc_reset_upload_document_body(request: UploadResetRequest):
    from tools import reset_uploaded_document

    reset_uploaded_document(_normalize_upload_session_id(request.effective_session_id()))
    return {"ok": True}


async def _collect_streamed_tech_doc_answer(request: ChatRequest) -> str:
    from routes.tech_doc import run_tech_doc_graph_stream

    chunks = []
    async for chunk in run_tech_doc_graph_stream(
        request.question,
        request.history,
        upload_session_id=request.effective_upload_session_id(),
    ):
        event = json.loads(chunk)
        if event.get("type") == "token":
            chunks.append(event.get("content", ""))
        elif event.get("type") == "done":
            break
    return "".join(chunks).strip()


@app.post("/api/tech-doc/chat")
async def tech_doc_chat(request: ChatRequest):
    from routes.tech_doc import run_tech_doc_graph

    try:
        result = await run_tech_doc_graph(
            request.question,
            request.history,
            upload_session_id=request.effective_upload_session_id(),
        )
        answer = result["generation"]
    except Exception:
        answer = await _collect_streamed_tech_doc_answer(request)
        if not answer:
            raise
    return ChatResponse(answer=answer)


@app.post("/api/tech-doc/chat/stream")
async def tech_doc_chat_stream(request: ChatRequest):
    from routes.tech_doc import run_tech_doc_graph_stream

    async def generate():
        async for chunk in run_tech_doc_graph_stream(
            request.question,
            request.history,
            upload_session_id=request.effective_upload_session_id(),
        ):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
