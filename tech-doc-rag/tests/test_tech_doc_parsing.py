from pathlib import Path
import sys
import json
import asyncio


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api" / "routes"))

from tech_doc import _parse_generation, run_tech_doc_graph, run_tech_doc_graph_stream


def collect_stream_events():
    async def collect_events():
        events = []
        async for chunk in run_tech_doc_graph_stream("hello"):
            events.append(json.loads(chunk))
        return events

    return asyncio.run(collect_events())


def test_parse_generation_with_xml_tags():
    generation = """<reasoning>
The docs describe StateGraph as the workflow abstraction.
</reasoning>
<answer>
StateGraph is LangGraph's graph abstraction for defining workflow state and transitions.
</answer>"""

    reasoning, answer = _parse_generation(generation)

    assert reasoning == "The docs describe StateGraph as the workflow abstraction."
    assert answer == "StateGraph is LangGraph's graph abstraction for defining workflow state and transitions."


def test_parse_generation_with_legacy_markers():
    generation = "Reasoning: Check the retrieved docs first. Answer: Hello!"

    reasoning, answer = _parse_generation(generation)

    assert reasoning == "Check the retrieved docs first."
    assert answer == "Hello!"


def test_parse_generation_with_missing_answer_closing_tag():
    generation = """<reasoning>
Plan the code example.
</reasoning>
<answer>
```python
print("Hello")
```"""

    reasoning, answer = _parse_generation(generation)

    assert reasoning == "Plan the code example."
    assert answer == '```python\nprint("Hello")\n```'


def test_parse_generation_strips_leaked_xml_tags_inside_sections():
    generation = """<reasoning>
<reasoning>Plan.</reasoning>
</reasoning>
<answer>
<answer>Hello!</answer>
</answer>"""

    reasoning, answer = _parse_generation(generation)

    assert reasoning == "Plan."
    assert answer == "Hello!"


def test_parse_generation_ignores_answer_tag_mentions_inside_reasoning():
    generation = """<reasoning>
The response must put final text inside <answer> tags.
</reasoning>
<answer>
```python
print("Hello")
```
</answer>"""

    reasoning, answer = _parse_generation(generation)

    assert "final text inside" in reasoning
    assert "<answer>" not in reasoning
    assert answer == '```python\nprint("Hello")\n```'


def test_parse_generation_strips_unreadable_line_citation_artifacts():
    artifact = "\u30106\u2020L1-L9\u3011"
    generation = """<reasoning>
Use retrieved docs.
</reasoning>
<answer>
StateGraph controls workflow state and routing {artifact}. Use readable citations instead. [langgraph/overview.md]
</answer>""".format(artifact=artifact)

    reasoning, answer = _parse_generation(generation)

    assert reasoning == "Use retrieved docs."
    assert artifact not in answer
    assert answer.endswith("[langgraph/overview.md]")


def test_parse_generation_strips_trailing_source_list():
    generation = """<reasoning>
Use retrieved docs.
</reasoning>
<answer>
StateGraph uses nodes and edges. [langgraph/overview.md]

Sources:
- [langgraph/overview.md]
- [langsmith/overview.md]
</answer>"""

    reasoning, answer = _parse_generation(generation)

    assert reasoning == "Use retrieved docs."
    assert answer == "StateGraph uses nodes and edges. [langgraph/overview.md]"
    assert "Sources:" not in answer


def test_stream_emits_thinking_and_done(monkeypatch):
    async def fake_run_tech_doc_graph(question, history=None):
        return {
            "generation": """<reasoning>Check the docs.</reasoning><answer>Here is the final answer.</answer>"""
        }

    monkeypatch.setattr("tech_doc.run_tech_doc_graph", fake_run_tech_doc_graph)

    events = collect_stream_events()

    assert events[0]["type"] == "system_prompt"
    assert events[1]["type"] == "user_prompt"
    assert any(event["type"] == "thinking_done" for event in events)
    assert any(event["type"] == "token" for event in events)
    assert events[-1]["type"] == "done"


def test_stream_respects_configured_chunk_size(monkeypatch):
    import config

    async def fake_run_tech_doc_graph(question, history=None):
        return {
            "generation": """<reasoning>abcdefghi</reasoning><answer>jklmnopqr</answer>"""
        }

    monkeypatch.setattr("tech_doc.run_tech_doc_graph", fake_run_tech_doc_graph)
    monkeypatch.setattr(config.settings, "stream_chunk_size", 3)

    events = collect_stream_events()
    streamed_chunks = [event["content"] for event in events if event["type"] == "token"]

    assert streamed_chunks
    assert all(len(chunk) <= 3 for chunk in streamed_chunks)


def test_run_tech_doc_graph_falls_back_to_last_plain_ai_message(monkeypatch):
    class AIMessage:
        content = "Fallback final answer"
        tool_calls = []

    def fake_invoke(state, config):
        return {"generation": "", "messages": [AIMessage()], "trace": [{"step": "agent_generating"}]}

    monkeypatch.setattr("tech_doc.tech_doc_graph.invoke", fake_invoke)

    result = asyncio.run(run_tech_doc_graph("hello"))

    assert result == {
        "generation": "Fallback final answer",
        "trace": [{"step": "agent_generating"}],
    }
