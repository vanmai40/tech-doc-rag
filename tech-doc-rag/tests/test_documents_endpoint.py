from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api"))

from main import _load_tech_doc_documents, app


def test_tech_doc_documents_endpoint_returns_raw_markdown_sources():
    client = TestClient(app)

    response = client.get("/api/tech-doc/documents")

    assert response.status_code == 200
    documents = response.json()["documents"]
    document_ids = {document["id"] for document in documents}
    assert {"langgraph/overview.md", "langsmith/overview.md"}.issubset(document_ids)

    langgraph = next(document for document in documents if document["id"] == "langgraph/overview.md")
    assert langgraph["title"] == "LangGraph Overview"
    assert langgraph["source"] == "langgraph"
    assert langgraph["path"] == "langgraph/overview.md"
    assert "# LangGraph Overview" in langgraph["content"]
    assert "ReAct Agents and Tool Calling" in langgraph["content"]
    assert "Persistence and Checkpointing" in langgraph["content"]
    assert langgraph["size_bytes"] == len(langgraph["content"].encode("utf-8"))
    assert langgraph["line_count"] == len(langgraph["content"].splitlines())

    langsmith = next(document for document in documents if document["id"] == "langsmith/overview.md")
    assert "Debugging Agents" in langsmith["content"]
    assert "Evaluating Agent Workflows" in langsmith["content"]


def test_load_tech_doc_documents_restricts_to_markdown_inside_data_dir(tmp_path):
    docs_dir = tmp_path / "data"
    docs_dir.mkdir()
    (docs_dir / "visible.md").write_text("# Visible\n\nAllowed", encoding="utf-8")
    (docs_dir / "hidden.txt").write_text("Not markdown", encoding="utf-8")

    documents = _load_tech_doc_documents(docs_dir)

    assert [document["id"] for document in documents] == ["visible.md"]
    assert documents[0]["content"] == "# Visible\n\nAllowed"


def test_upload_document_endpoint_returns_uploaded_document_metadata(monkeypatch):
    pytest.importorskip("multipart")

    expected = {
        "id": "uploaded:session-1:notes.md",
        "title": "Uploaded Notes",
        "source": "uploaded",
        "path": "notes.md",
        "content": "# Uploaded Notes\n\nTemporary content",
        "size_bytes": 35,
        "line_count": 3,
        "session_id": "session-1",
        "uploaded": True,
    }

    def fake_ingest(session_id, filename, content_type, data):
        assert session_id == "session-1"
        assert filename == "notes.md"
        assert content_type == "text/markdown"
        assert data == b"# Uploaded Notes\n\nTemporary content"
        return expected

    monkeypatch.setattr("tools.ingest_uploaded_document", fake_ingest)
    client = TestClient(app)

    response = client.post(
        "/api/tech-doc/documents/upload",
        data={"session_id": "session-1"},
        files={"file": ("notes.md", b"# Uploaded Notes\n\nTemporary content", "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json()["document"] == expected


def test_upload_reset_endpoint_removes_session_document(monkeypatch):
    reset_calls = []

    def fake_reset(session_id):
        reset_calls.append(session_id)

    monkeypatch.setattr("tools.reset_uploaded_document", fake_reset)
    client = TestClient(app)

    response = client.delete("/api/tech-doc/documents/upload/session-1")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert reset_calls == ["session-1"]
