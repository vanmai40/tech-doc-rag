import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document


def test_ingest_uploaded_document_builds_session_vectorstore(monkeypatch):
    import tools

    captured = {}

    class FakeVectorstore:
        def __init__(self, docs):
            self.docs = docs

        @classmethod
        def from_documents(cls, docs, embeddings):
            captured["docs"] = docs
            captured["embeddings"] = embeddings
            return cls(docs)

        def similarity_search(self, query, k=3):
            return self.docs[:k]

        def similarity_search_with_score(self, query, k=3):
            return [(doc, float(index)) for index, doc in enumerate(self.docs[:k])]

        def merge_from(self, other):
            self.docs.extend(other.docs)

    monkeypatch.setattr(tools, "FAISS", FakeVectorstore)
    monkeypatch.setattr(tools, "get_embeddings", lambda: object())
    monkeypatch.setattr(tools, "get_vectorstore", lambda: (_ for _ in ()).throw(FileNotFoundError()))
    tools.reset_uploaded_document("session-1")

    row = tools.ingest_uploaded_document(
        "session-1",
        "notes.md",
        "text/markdown",
        b"# Uploaded Notes\n\nThis uploaded file explains temporary retrieval.",
    )

    assert row["id"] == "uploaded:session-1:notes.md"
    assert row["title"] == "Uploaded Notes"
    assert row["uploaded"] is True
    assert captured["docs"]
    assert captured["docs"][0].metadata["source"] == "uploaded"

    retrieved = tools.retrieve_documents("temporary retrieval", upload_session_id="session-1")
    assert retrieved
    assert retrieved[0].metadata["path"] == "notes.md"

    tools.reset_uploaded_document("session-1")
    assert tools.get_uploaded_document("session-1") is None


def test_retrieve_documents_queries_combined_session_vectorstore(monkeypatch):
    import tools

    static_best = Document(page_content="Static pricing match", metadata={"source": "langgraph", "path": "langgraph/overview.md"})
    static_weak = Document(page_content="Weak static match", metadata={"source": "langsmith", "path": "langsmith/overview.md"})
    uploaded_best = Document(page_content="Uploaded Z.AI pricing", metadata={"source": "uploaded", "path": "pricing.md"})
    uploaded_weak = Document(page_content="Weak uploaded match", metadata={"source": "uploaded", "path": "pricing.md"})

    class FakeVectorstore:
        def __init__(self, results):
            self.results = results

        def similarity_search_with_score(self, query, k=3):
            return self.results[:k]

    with tools._uploaded_lock:
        tools._session_vectorstores["session-merge"] = FakeVectorstore([
            (static_weak, 0.8),
            (static_best, 0.1),
            (uploaded_best, 0.05),
            (uploaded_weak, 0.9),
        ])

    try:
        docs = tools.retrieve_documents("z.ai pricing", top_k=3, upload_session_id="session-merge")
    finally:
        tools.reset_uploaded_document("session-merge")

    assert [doc.page_content for doc in docs] == [
        "Weak static match",
        "Static pricing match",
        "Uploaded Z.AI pricing",
    ]


def test_ingest_uploaded_document_builds_combined_session_vectorstore(monkeypatch):
    import tools

    uploaded_doc = Document(page_content="Uploaded pricing", metadata={"source": "uploaded", "path": "pricing.md"})
    static_doc = Document(page_content="Static LangGraph", metadata={"source": "langgraph", "path": "langgraph/overview.md"})

    class FakeVectorstore:
        def __init__(self, docs):
            self.docs = list(docs)

        @classmethod
        def from_documents(cls, docs, embeddings):
            return cls(docs)

        def merge_from(self, other):
            self.docs.extend(other.docs)

        def similarity_search_with_score(self, query, k=3):
            return [(doc, float(index)) for index, doc in enumerate(self.docs[:k])]

    monkeypatch.setattr(tools, "FAISS", FakeVectorstore)
    monkeypatch.setattr(tools, "get_embeddings", lambda: object())
    monkeypatch.setattr(tools, "get_vectorstore", lambda: FakeVectorstore([static_doc]))

    tools.ingest_uploaded_document("session-combined", "pricing.md", "text/markdown", uploaded_doc.page_content.encode("utf-8"))
    try:
        docs = tools.retrieve_documents("pricing", top_k=2, upload_session_id="session-combined")
    finally:
        tools.reset_uploaded_document("session-combined")

    assert [doc.metadata["source"] for doc in docs] == ["uploaded", "langgraph"]


def test_format_retrieved_documents_includes_source_citation():
    from tools import format_retrieved_documents

    formatted = format_retrieved_documents([
        Document(
            page_content="StateGraph coordinates graph state.",
            metadata={"source": "langgraph", "path": "langgraph/overview.md"},
        )
    ])

    assert "[Document 1 | Source: langgraph/overview.md]" in formatted
    assert "Use this exact readable citation inline next to claims from this document: [langgraph/overview.md]" in formatted
    assert "StateGraph coordinates graph state." in formatted
