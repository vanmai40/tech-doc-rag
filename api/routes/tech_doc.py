import sys
import json
import asyncio
import re
import threading
from typing import AsyncGenerator, Optional
from pathlib import Path

tech_doc_src = Path(__file__).parent.parent.parent / "tech-doc-rag" / "src"
sys.path.insert(0, str(tech_doc_src))

from graph import graph as tech_doc_graph


SENSITIVE_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password|authorization|bearer)", re.IGNORECASE)
SECRET_VALUE_RE = re.compile(
    r"(?i)((?:api[_-]?key|token|secret|password|authorization)\s*[:=]\s*)(['\"]?)[^\s,'\"]+\2"
)


def _redact_and_limit_text(text: object, limit: int) -> str:
    value = "" if text is None else str(text)
    value = SECRET_VALUE_RE.sub(r"\1[redacted]", value)
    if len(value) > limit:
        value = value[:limit].rstrip() + "... [truncated]"
    return value


def _sanitize_trace_value(value, limit: Optional[int] = None):
    import config

    text_limit = limit or config.settings.trace_text_limit
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if SENSITIVE_KEY_RE.search(str(key)):
                sanitized[key] = "[redacted]"
            else:
                sanitized[key] = _sanitize_trace_value(item, text_limit)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_trace_value(item, text_limit) for item in value[:10]]
    if isinstance(value, str):
        return _redact_and_limit_text(value, text_limit)
    return value


def _strip_unreadable_citations(text: str) -> str:
    return re.sub(r"\s*【\d+†L\d+(?:-L?\d+)?】", "", text).strip()


def _strip_trailing_source_list(text: str) -> str:
    pattern = r"\n{1,3}(?:#{1,6}\s*)?(?:sources|references|bibliography)\s*:?\s*\n[\s\S]*$"
    return re.sub(pattern, "", text.strip(), flags=re.IGNORECASE).strip()


def _clean_answer(answer: str) -> str:
    return _strip_trailing_source_list(_strip_unreadable_citations(answer))


async def run_tech_doc_graph(
    question: str,
    history: list[dict] = None,
    upload_session_id: Optional[str] = None,
) -> dict:
    result = tech_doc_graph.invoke(
        {
            "question": question,
            "messages": [],
            "generation": "",
            "sources": [],
            "history": history or [],
            "upload_session_id": upload_session_id,
            "iteration_count": 0,
            "trace": [],
        },
        {"recursion_limit": 25},
    )

    generation = result.get("generation", "") or next(
        (
            msg.content
            for msg in reversed(result.get("messages", []))
            if type(msg).__name__ == "AIMessage"
            and getattr(msg, "content", "")
            and not getattr(msg, "tool_calls", None)
        ),
        "",
    )

    return {"generation": generation, "trace": result.get("trace", [])}


def _parse_generation(generation: str) -> tuple[str, str]:
    reasoning = ""
    answer = generation

    reasoning_match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", generation, flags=re.IGNORECASE | re.DOTALL)
    answer_source = generation[reasoning_match.end():] if reasoning_match else generation
    answer_match = re.search(r"<answer>\s*(.*?)(?:\s*</answer>|\s*$)", answer_source, flags=re.IGNORECASE | re.DOTALL)
    if reasoning_match or answer_match:
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        answer = answer_match.group(1).strip() if answer_match else ""
        reasoning = re.sub(r"</?(?:reasoning|answer)>", "", reasoning, flags=re.IGNORECASE).strip()
        answer = re.sub(r"</?(?:reasoning|answer)>", "", answer, flags=re.IGNORECASE).strip()
        answer = _clean_answer(answer)
        return reasoning, answer

    lower = generation.lower()

    reasoning_idx = lower.find("reasoning:")
    answer_idx = lower.find("answer:")

    if reasoning_idx != -1 and answer_idx != -1 and answer_idx > reasoning_idx:
        reasoning = generation[reasoning_idx + len("reasoning:"):answer_idx].strip()
        answer = generation[answer_idx + len("answer:"):].strip()
    elif reasoning_idx != -1:
        reasoning = generation[reasoning_idx + len("reasoning:"):].strip()
        answer = ""
    else:
        markers = ["## answer", "**answer**", "answer\n", "final answer"]
        for marker in markers:
            idx = lower.find(marker)
            if idx != -1:
                reasoning = generation[:idx].strip()
                answer = generation[idx + len(marker):].strip()
                if reasoning.lower().startswith("reasoning"):
                    reasoning = reasoning[len("reasoning"):].lstrip(":-*_ \n")
                break

    if not reasoning and answer.strip() == generation.strip():
        reasoning_prefixes = (
            "the user asks",
            "the user wants",
            "i need to",
            "i should",
            "we need to",
        )
        stripped = generation.strip()
        if stripped.lower().startswith(reasoning_prefixes):
            sentence_end = re.search(r"(?<=[.!?])\s+(?=[A-Z])", stripped)
            if sentence_end:
                reasoning = stripped[:sentence_end.start()].strip()
                answer = stripped[sentence_end.end():].strip()
            else:
                answer = stripped

    return reasoning, _clean_answer(answer)


def _stream_text(text: str, chunk_size: int = 7):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


async def run_tech_doc_graph_stream(
    question: str,
    history: list[dict] = None,
    upload_session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    import prompts as _prompts
    import config

    delay = config.settings.stream_chunk_delay

    if not history:
        yield json.dumps({"type": "system_prompt", "content": _sanitize_trace_value(_prompts.REACT_SYSTEM_PROMPT)})
        await asyncio.sleep(delay)
    yield json.dumps({"type": "user_prompt", "content": _sanitize_trace_value(question)})
    await asyncio.sleep(delay)

    chunk_size = config.settings.stream_chunk_size
    state = {
        "question": question,
        "messages": [],
        "generation": "",
        "sources": [],
        "history": history or [],
        "upload_session_id": upload_session_id,
        "iteration_count": 0,
        "trace": [],
    }
    graph_config = {"recursion_limit": 25}
    seen_trace = 0
    generation = ""
    trace = []

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def run_graph_stream():
        try:
            for item in tech_doc_graph.stream(state, graph_config, stream_mode="updates"):
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"__error__": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=run_graph_stream, daemon=True).start()

    while True:
        update = await queue.get()
        if update is None:
            break
        if "__error__" in update:
            yield json.dumps({"type": "token", "content": "The demo runtime hit an error while answering. Please try again with a shorter or simpler question."})
            yield json.dumps({"type": "done"})
            return
        for node_update in update.values():
            if not isinstance(node_update, dict):
                continue
            if "trace" in node_update:
                trace = node_update.get("trace", trace)
                for step in trace[seen_trace:]:
                    trace_id = step.get("trace_id", 0)
                    if step["step"] in {"llm_reasoning", "llm_response"}:
                        yield json.dumps({
                            "type": "llm_response",
                            "trace_id": trace_id,
                            "content": _sanitize_trace_value(step.get("content", "")),
                        })
                        await asyncio.sleep(delay)
                    elif step["step"] == "agent_decided":
                        yield json.dumps({
                            "type": "tool_call",
                            "trace_id": trace_id,
                            "tool": step.get("tool", "unknown"),
                            "args": _sanitize_trace_value(step.get("args", {})),
                        })
                        await asyncio.sleep(delay)
                    elif step["step"] == "tool_result":
                        yield json.dumps({
                            "type": "tool_result",
                            "trace_id": trace_id,
                            "tool": step.get("tool", "unknown"),
                            "preview": _sanitize_trace_value(step.get("preview", "")),
                            "result": _sanitize_trace_value(
                                step.get("result", step.get("preview", "")),
                                limit=config.settings.trace_result_limit,
                            ),
                        })
                        await asyncio.sleep(delay)
                seen_trace = len(trace)
            if "generation" in node_update:
                generation = node_update.get("generation", generation)

    reasoning, answer = _parse_generation(generation)

    if reasoning:
        for chunk in _stream_text(reasoning, chunk_size=chunk_size):
            yield json.dumps({"type": "thinking", "content": chunk})
            await asyncio.sleep(delay)
    yield json.dumps({"type": "thinking_done"})

    if not answer.strip():
        failed_tools = [
            step for step in trace
            if step.get("step") == "tool_result" and (
                "rate-limited" in step.get("preview", "").lower()
                or "no search results" in step.get("preview", "").lower()
                or "search error" in step.get("preview", "").lower()
            )
        ]
        if failed_tools:
            answer = (
                "I could not complete the web lookup because the search provider returned "
                "rate-limit/no-result responses. Please try again in a moment, or provide a "
                "specific source URL and I can use that context."
            )
        else:
            answer = "I could not produce an answer from the available tool results."

    for chunk in _stream_text(answer, chunk_size=chunk_size):
        yield json.dumps({"type": "token", "content": chunk})
        await asyncio.sleep(delay)

    yield json.dumps({"type": "done"})
