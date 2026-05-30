import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from graph import (
    GraphState,
    _build_messages,
    _mock_agent_response,
    agent_node,
    router,
    execute_tool,
    extract_generation,
)
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage


class TestMockAgentResponse:
    def test_returns_react_answer(self):
        state: GraphState = {
            "question": "What is LangGraph?",
            "messages": [],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        }
        response = _mock_agent_response(state)
        assert "<reasoning>" in response.content
        assert "<answer>" in response.content
        assert "LangGraph" in response.content or "mock" in response.content.lower()

    def test_includes_question_in_response(self):
        state: GraphState = {
            "question": "How does StateGraph work?",
            "messages": [],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        }
        response = _mock_agent_response(state)
        assert "StateGraph" in response.content or "How does" in response.content


class TestBuildMessages:
    def test_uses_full_system_prompt_on_first_turn(self):
        state: GraphState = {
            "question": "What is LangGraph?",
            "messages": [],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        }

        messages = _build_messages(state)

        assert "You are a helpful technical documentation assistant" in str(messages[0].content)
        assert "search_web(query)" in str(messages[0].content)

    def test_uses_full_system_prompt_on_followup(self):
        state: GraphState = {
            "question": "How does routing work?",
            "messages": [],
            "generation": "",
            "sources": [],
            "history": [{"role": "user", "content": "What is LangGraph?"}],
            "iteration_count": 0,
            "trace": [],
        }

        messages = _build_messages(state)

        assert "You are a helpful technical documentation assistant" in str(messages[0].content)
        assert "search_web(query)" in str(messages[0].content)

    def test_adds_uploaded_document_tool_guidance(self, monkeypatch):
        monkeypatch.setattr(
            "graph.get_uploaded_document",
            lambda session_id: {"title": "Uploaded Notes", "path": "notes.md"},
        )
        state: GraphState = {
            "question": "What does this file say about zebracite?",
            "messages": [],
            "generation": "",
            "sources": [],
            "history": [],
            "upload_session_id": "session-1",
            "iteration_count": 0,
            "trace": [],
        }

        messages = _build_messages(state)
        guidance = "\n".join(str(message.content) for message in messages)

        assert "A temporary uploaded document is active" in guidance
        assert "Uploaded Notes" in guidance
        assert "call retrieve_docs first" in guidance
        assert "Do not call search_web for uploaded-document questions" in guidance

    def test_followup_with_upload_uses_full_prompt_plus_upload_guidance(self, monkeypatch):
        monkeypatch.setattr(
            "graph.get_uploaded_document",
            lambda session_id: {"title": "Uploaded Notes", "path": "notes.md"},
        )
        state: GraphState = {
            "question": "What does this new upload say?",
            "messages": [],
            "generation": "",
            "sources": [],
            "history": [{"role": "user", "content": "Earlier turn before upload"}],
            "upload_session_id": "session-1",
            "iteration_count": 0,
            "trace": [],
        }

        messages = _build_messages(state)
        guidance = "\n".join(str(message.content) for message in messages[:2])

        assert "You are a helpful technical documentation assistant" in str(messages[0].content)
        assert "A temporary uploaded document is active" in guidance
        assert "Uploaded Notes" in guidance


class TestAgentNode:
    def test_active_upload_overrides_first_web_search_tool(self, monkeypatch):
        class ToolResponse:
            content = ""
            tool_calls = [{"name": "search_web", "args": {"query": "zebracite"}, "id": "call_web", "type": "tool_call"}]

        web_response = ToolResponse()

        class FakeLlmWithTools:
            def invoke(self, messages):
                return web_response

        class FakeLlm:
            def bind_tools(self, tools):
                return FakeLlmWithTools()

        monkeypatch.setattr("graph.get_uploaded_document", lambda session_id: {"title": "Uploaded Notes", "path": "notes.md"})
        monkeypatch.setattr("graph.get_llm", lambda: FakeLlm())
        monkeypatch.setattr("graph.settings.mock_llm", False)

        state: GraphState = {
            "question": "What does zebracite mean in this document?",
            "messages": [],
            "generation": "",
            "sources": [],
            "history": [{"role": "user", "content": "Earlier chat before upload"}],
            "upload_session_id": "session-1",
            "iteration_count": 0,
            "trace": [],
        }

        result = agent_node(state)
        response = result["messages"][0]

        assert response.tool_calls[0]["name"] == "retrieve_docs"
        assert response.tool_calls[0]["args"] == {"query": "What does zebracite mean in this document?"}


class TestRouter:
    def test_route_to_tool_when_tool_calls_present(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "search_web", "args": {"query": "test"}, "id": "1", "type": "tool_call"}],
        )
        state: GraphState = {
            "question": "test",
            "messages": [msg],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        }
        assert router(state) == "execute_tool"

    def test_route_to_end_when_no_tool_calls(self):
        msg = AIMessage(content="Final answer here")
        state: GraphState = {
            "question": "test",
            "messages": [msg],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        }
        assert router(state) == "end"

    def test_route_to_end_when_max_iterations_reached(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "search_web", "args": {"query": "test"}, "id": "1", "type": "tool_call"}],
        )
        state: GraphState = {
            "question": "test",
            "messages": [msg],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 10,  # max_tool_iterations in config
            "trace": [],
        }
        assert router(state) == "end"

    def test_route_to_end_with_empty_content_no_tool_calls(self):
        msg = AIMessage(content="")
        state: GraphState = {
            "question": "test",
            "messages": [msg],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        }
        assert router(state) == "end"


class TestExecuteTool:
    def test_returns_empty_if_no_tool_calls(self):
        msg = AIMessage(content="Just a response")
        state: GraphState = {
            "question": "test",
            "messages": [msg],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        }
        result = execute_tool(state)
        assert result["messages"] == []

    def test_returns_tool_message_for_unknown_tool(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "nonexistent_tool", "args": {}, "id": "call_1", "type": "tool_call"}],
        )
        state: GraphState = {
            "question": "test",
            "messages": [msg],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        }
        result = execute_tool(state)
        assert len(result["messages"]) == 1
        tm = result["messages"][0]
        assert isinstance(tm, ToolMessage)
        assert "Unknown tool" in tm.content

    def test_retrieve_docs_passes_upload_session_id(self, monkeypatch):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "retrieve_docs", "args": {"query": "temporary docs"}, "id": "call_1", "type": "tool_call"}],
        )
        calls = []

        def fake_retrieve_documents(query, top_k=3, upload_session_id=None):
            calls.append((query, top_k, upload_session_id))
            return []

        monkeypatch.setattr("graph.retrieve_documents", fake_retrieve_documents)
        state: GraphState = {
            "question": "test",
            "messages": [msg],
            "generation": "",
            "sources": [],
            "history": [],
            "upload_session_id": "session-1",
            "iteration_count": 0,
            "trace": [],
        }

        result = execute_tool(state)

        assert calls == [("temporary docs", 3, "session-1")]
        assert "No relevant documents" in result["messages"][0].content

    def test_retrieve_docs_formats_inline_citation_source(self, monkeypatch):
        from langchain_core.documents import Document

        msg = AIMessage(
            content="",
            tool_calls=[{"name": "retrieve_docs", "args": {"query": "temporary docs"}, "id": "call_1", "type": "tool_call"}],
        )

        def fake_retrieve_documents(query, top_k=3, upload_session_id=None):
            return [
                Document(
                    page_content="Uploaded details for retrieval.",
                    metadata={"source": "uploaded", "path": "notes.md"},
                )
            ]

        monkeypatch.setattr("graph.retrieve_documents", fake_retrieve_documents)
        state: GraphState = {
            "question": "test",
            "messages": [msg],
            "generation": "",
            "sources": [],
            "history": [],
            "upload_session_id": "session-1",
            "iteration_count": 0,
            "trace": [],
        }

        result = execute_tool(state)

        content = result["messages"][0].content
        assert "[Document 1 | Source: uploaded/notes.md]" in content
        assert "Use this exact readable citation inline next to claims from this document: [uploaded/notes.md]" in content


class TestExtractGeneration:
    def test_extracts_from_last_ai_message(self):
        state: GraphState = {
            "question": "test",
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_web", "args": {"query": "test"}, "id": "1", "type": "tool_call"}],
                ),
                ToolMessage(content="Search results here", tool_call_id="1"),
                AIMessage(content="<reasoning>My reasoning</reasoning>\n<answer>Final answer</answer>"),
            ],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        }
        result = extract_generation(state)
        assert "Final answer" in result["generation"]

    def test_returns_empty_if_no_ai_message(self):
        state: GraphState = {
            "question": "test",
            "messages": [HumanMessage(content="hello")],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        }
        result = extract_generation(state)
        assert result["generation"] == ""


class TestGraphEndToEnd:
    def test_graph_runs_with_mock_llm(self):
        from graph import graph

        result = graph.invoke({
            "question": "What is LangGraph?",
            "messages": [],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        })
        assert "generation" in result
        assert "mock" in result["generation"].lower()

    def test_graph_includes_trace(self):
        from graph import graph

        result = graph.invoke({
            "question": "Hello",
            "messages": [],
            "generation": "",
            "sources": [],
            "history": [],
            "iteration_count": 0,
            "trace": [],
        })
        assert "trace" in result
        assert len(result["trace"]) > 0

    def test_graph_respects_history(self):
        from graph import graph

        result = graph.invoke({
            "question": "What is LangGraph?",
            "messages": [],
            "generation": "",
            "sources": [],
            "history": [{"role": "user", "content": "Previous question"}, {"role": "assistant", "content": "Previous answer"}],
            "iteration_count": 0,
            "trace": [],
        })
        assert "generation" in result
        assert len(result["generation"]) > 0
