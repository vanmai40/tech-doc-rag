"""
Regression test suite for tech-doc-rag.

Run with: pytest tests/ -v
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDocumentIngestion:
    """Tests for document loading and chunking."""

    def test_load_documents_from_data_dir(self):
        from ingest import load_documents
        data_dir = Path(__file__).parent.parent / "data"
        if not data_dir.exists():
            pytest.skip("No data directory found")

        documents = load_documents(str(data_dir))
        assert len(documents) > 0
        assert all(hasattr(doc, 'page_content') for doc in documents)

    def test_chunk_documents(self):
        from ingest import chunk_documents
        from langchain_core.documents import Document

        docs = [Document(page_content="This is a test document. " * 100)]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 0
        assert all(hasattr(chunk, 'page_content') for chunk in chunks)

    def test_chunk_size_respected(self):
        from ingest import chunk_documents
        from langchain_core.documents import Document

        docs = [Document(page_content="word " * 200)]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=0)

        assert len(chunks) >= 2

    def test_default_docs_are_substantial(self):
        data_dir = Path(__file__).parent.parent / "data"
        langgraph = (data_dir / "langgraph" / "overview.md").read_text(encoding="utf-8")
        langsmith = (data_dir / "langsmith" / "overview.md").read_text(encoding="utf-8")

        assert len(langgraph.splitlines()) > 250
        assert "Conditional Edges" in langgraph
        assert "Human-in-the-Loop Workflows" in langgraph
        assert len(langsmith.splitlines()) > 220
        assert "Datasets" in langsmith
        assert "Monitoring Production Applications" in langsmith


class TestVectorStore:
    """Tests for vector store operations."""

    def test_faiss_index_creation(self):
        from ingest import get_embeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        embeddings = get_embeddings()
        docs = [
            Document(page_content="LangGraph is a library for building stateful apps"),
            Document(page_content="LangSmith is for debugging LLM applications")
        ]

        vectorstore = FAISS.from_documents(docs, embeddings)
        assert vectorstore is not None

        # Test similarity search
        results = vectorstore.similarity_search("What is LangGraph?", k=1)
        assert len(results) > 0
        assert "LangGraph" in results[0].page_content

    def test_faiss_save_and_load(self, tmp_path):
        from ingest import get_embeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        embeddings = get_embeddings()
        docs = [Document(page_content="Test document content")]

        vectorstore = FAISS.from_documents(docs, embeddings)
        index_path = tmp_path / "test_index"
        vectorstore.save_local(str(index_path))

        loaded = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
        assert loaded is not None


class TestEmbeddings:
    """Tests for embedding models."""

    def test_huggingface_embeddings(self):
        from ingest import get_embeddings

        embeddings = get_embeddings()
        vector = embeddings.embed_query("test query")

        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(x, float) for x in vector)

    def test_embedding_dimensions(self):
        from ingest import get_embeddings

        embeddings = get_embeddings()
        vector = embeddings.embed_query("test")

        # all-MiniLM-L6-v2 produces 384-dimensional vectors
        assert len(vector) == 384


class TestGraphNodes:
    """Tests for ReAct graph node functions."""

    def test_agent_node_returns_dict(self):
        from graph import agent_node, GraphState

        result = agent_node({
            "question": "test",
            "messages": [],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        })
        assert isinstance(result, dict)
        assert "messages" in result
        assert "trace" in result
        assert len(result["messages"]) == 1
        # Should be an AI message (from mock_llm)
        assert "mock" in result["messages"][0].content.lower()

    def test_router_function(self):
        from graph import router, GraphState
        from langchain_core.messages import AIMessage

        # No tool calls -> end
        state: GraphState = {
            "question": "test",
            "messages": [AIMessage(content="Hello")],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        }
        assert router(state) == "end"

    def test_execute_tool_dispatches_by_name(self):
        from graph import execute_tool, GraphState
        from langchain_core.messages import AIMessage

        # Unknown tool -> error message
        state: GraphState = {
            "question": "test",
            "messages": [AIMessage(
                content="",
                tool_calls=[{"name": "nonexistent", "args": {}, "id": "call_1", "type": "tool_call"}],
            )],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        }
        result = execute_tool(state)
        assert "Unknown tool" in result["messages"][0].content


class TestGraphState:
    """Tests for graph state management."""

    def test_state_typeddict(self):
        from graph import GraphState

        assert hasattr(GraphState, '__annotations__')
        annotations = GraphState.__annotations__
        assert 'question' in annotations
        assert 'messages' in annotations
        assert 'generation' in annotations
        assert 'trace' in annotations
        assert 'history' in annotations

    def test_state_initialization(self):
        from graph import GraphState

        state = GraphState(
            question="What is LangGraph?",
            messages=[],
            generation="",
            sources=[],
            history=[],
            iteration_count=0,
            trace=[],
        )
        assert state['question'] == "What is LangGraph?"
        assert state['iteration_count'] == 0
        assert state['trace'] == []


class TestTools:
    """Tests for tool functions."""

    def test_get_embeddings_provider(self):
        from tools import get_embeddings

        # Test that get_embeddings returns a valid embeddings object
        embeddings = get_embeddings()
        assert embeddings is not None

    @patch('tools.get_vectorstore')
    def test_retrieve_documents_mock(self, mock_get_vectorstore):
        from tools import retrieve_documents
        from langchain_core.documents import Document

        mock_vs = Mock()
        mock_vs.similarity_search.return_value = [
            Document(page_content="Test result")
        ]
        mock_get_vectorstore.return_value = mock_vs

        results = retrieve_documents("test query")
        assert len(results) == 1
        assert results[0].page_content == "Test result"


class TestPrompts:
    """Tests for ReAct system prompt."""

    def test_react_system_prompt_exists(self):
        from prompts import REACT_SYSTEM_PROMPT
        assert len(REACT_SYSTEM_PROMPT) > 100
        assert "search_web" in REACT_SYSTEM_PROMPT
        assert "fetch_url" in REACT_SYSTEM_PROMPT
        assert "retrieve_docs" in REACT_SYSTEM_PROMPT
        assert "run_python" in REACT_SYSTEM_PROMPT

    def test_react_prompt_has_tool_descriptions(self):
        from prompts import REACT_SYSTEM_PROMPT
        assert "search the web" in REACT_SYSTEM_PROMPT.lower()
        assert "knowledge base" in REACT_SYSTEM_PROMPT.lower()
        assert "sandboxed" in REACT_SYSTEM_PROMPT.lower()

    def test_react_prompt_enforces_cot_contract(self):
        from prompts import REACT_SYSTEM_PROMPT
        assert "<reasoning>" in REACT_SYSTEM_PROMPT
        assert "</reasoning>" in REACT_SYSTEM_PROMPT
        assert "<answer>" in REACT_SYSTEM_PROMPT
        assert "</answer>" in REACT_SYSTEM_PROMPT


class TestConfig:
    """Tests for configuration."""

    def test_config_values_exist(self):
        from config import settings
        assert settings.max_retries == 2
        assert settings.top_k == 3

    def test_llm_config_defaults(self):
        from config import settings
        assert settings.llm_base_url is not None
        assert settings.llm_model is not None

    def test_streaming_config_regression_defaults(self):
        from config import settings

        assert settings.stream_chunk_size == 4
        assert settings.llm_max_tokens >= 4096


class TestIntegration:
    """Integration tests for end-to-end workflows."""

    @pytest.mark.skipif(not os.path.exists(".faiss_index"), reason="FAISS index not built")
    def test_full_retrieval_pipeline(self):
        from tools import retrieve_documents

        results = retrieve_documents("What is LangGraph?", top_k=2)
        assert len(results) <= 2
        assert all(hasattr(doc, 'page_content') for doc in results)

    def test_ingestion_pipeline(self, tmp_path):
        from ingest import load_documents, chunk_documents
        from langchain_community.vectorstores import FAISS
        from ingest import get_embeddings

        # Create test documents
        test_dir = tmp_path / "test_docs"
        test_dir.mkdir()
        (test_dir / "test.md").write_text("# Test\nThis is a test document about LangGraph.")

        docs = load_documents(str(test_dir))
        assert len(docs) == 1

        chunks = chunk_documents(docs)
        assert len(chunks) >= 1

        # Build FAISS index
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        assert vectorstore is not None


class TestAPI:
    """Tests for API endpoints."""

    @pytest.mark.asyncio
    async def test_tech_doc_route_exists(self):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api"))
        from routes.tech_doc import run_tech_doc_graph
        assert callable(run_tech_doc_graph)

    @pytest.mark.asyncio
    async def test_streaming_route_exists(self):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api"))
        from routes.tech_doc import run_tech_doc_graph_stream
        assert callable(run_tech_doc_graph_stream)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
