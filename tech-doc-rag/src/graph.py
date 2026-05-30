import json
from typing import TypedDict, Annotated, Optional
try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool

from config import settings
from tools import get_llm, REACT_TOOLS, search_web, fetch_url, retrieve_documents, format_retrieved_documents, get_uploaded_document, run_python
from prompts import REACT_SYSTEM_PROMPT


class GraphState(TypedDict):
    question: str
    messages: Annotated[list, add_messages]
    generation: str
    sources: list[dict]
    history: list[dict]
    upload_session_id: NotRequired[Optional[str]]
    iteration_count: int
    trace: list[dict]


def _build_messages(state: GraphState) -> list[BaseMessage]:
    """Build the message list for the LLM, including system prompt and history."""
    history = state.get("history", [])
    msgs = [SystemMessage(content=REACT_SYSTEM_PROMPT)]

    upload_session_id = state.get("upload_session_id")
    uploaded_document = get_uploaded_document(upload_session_id) if upload_session_id else None
    if uploaded_document:
        title = uploaded_document.get("title") or uploaded_document.get("path") or "uploaded document"
        path = uploaded_document.get("path") or "uploaded document"
        msgs.append(SystemMessage(content=(
            f"A temporary uploaded document is active for this chat: {title} ({path}). "
            "If the user's question could be answered from this uploaded document, call retrieve_docs first. "
            "Do not call search_web for uploaded-document questions unless retrieve_docs has no relevant result "
            "or the user explicitly asks for current web information."
        )))

    if history:
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content=content))
            else:
                msgs.append(AIMessage(content=content))

    existing = state.get("messages", [])
    has_human = any(isinstance(m, HumanMessage) for m in existing)
    if not has_human:
        msgs.append(HumanMessage(content=state["question"]))
    msgs.extend(existing)

    return msgs


def _mock_agent_response(state: GraphState) -> AIMessage:
    """Return a mock agent response for testing with mock_llm=True."""
    question = state["question"]
    content = (
        "<reasoning>\n"
        f"Mock reasoning for: {question}\n"
        "</reasoning>\n"
        "<answer>\n"
        f"This is a mock answer to: {question}\n"
        "</answer>"
    )
    return AIMessage(content=content)


_trace_counter: int = 0


def _next_trace_id() -> int:
    global _trace_counter
    _trace_counter += 1
    return _trace_counter


def _failed_tool_result(content: str) -> bool:
    lower = content.lower()
    return (
        "rate-limited by duckduckgo" in lower
        or "no search results found" in lower
        or "search error" in lower
    )


def _successful_tool_used(state: GraphState, tool_name: str) -> bool:
    for entry in state.get("trace", []):
        if entry.get("step") != "tool_result" or entry.get("tool") != tool_name:
            continue
        result = str(entry.get("result") or entry.get("preview") or "")
        if result and not _failed_tool_result(result):
            return True
    return False


def _technical_docs_question(question: str) -> bool:
    lower = question.lower()
    return any(term in lower for term in (
        "langgraph", "langsmith", "api", "docs", "documentation",
        "code example", "architecture", "python package", "class", "function",
    ))


def _current_web_question(question: str) -> bool:
    lower = question.lower()
    return any(term in lower for term in (
        "latest", "current", "recent", "news", "this week", "today",
        "online", "web", "internet", "search the web",
    ))


def _llm_response_log(response: AIMessage) -> str:
    content = response.content or ""
    if content.strip():
        return content
    tool_calls = getattr(response, "tool_calls", []) or []
    if tool_calls:
        return "Tool decision:\n" + json.dumps([
            {"tool": tc.get("name"), "args": tc.get("args", {})}
            for tc in tool_calls
        ], indent=2)
    return "LLM returned an empty response."


def agent_node(state: GraphState) -> dict:
    """Agent node: calls the LLM with tools bound, returns tool_calls or a final answer."""
    existing_messages = state.get("messages", [])
    if existing_messages and isinstance(existing_messages[-1], ToolMessage):
        tool_content = str(existing_messages[-1].content)
        if _failed_tool_result(tool_content):
            response = AIMessage(content=(
                "<answer>I could not complete the web lookup because the search provider "
                "returned a rate-limit/no-result response. Please try again in a moment, "
                "or provide a source URL and I can use that context.</answer>"
            ))
            return {
                "messages": [response],
                "iteration_count": state.get("iteration_count", 0) + 1,
                "trace": state.get("trace", []) + [
                    {"step": "llm_response", "trace_id": _next_trace_id(), "content": response.content},
                    {"step": "agent_generating", "trace_id": _next_trace_id(), "tool": "none", "preview": response.content[:200]}
                ],
            }

    if settings.mock_llm:
        response = _mock_agent_response(state)
        return {
            "messages": [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
            "trace": state.get("trace", []) + [
                {"step": "llm_response", "trace_id": _next_trace_id(), "content": response.content},
                {"step": "agent_generating", "tool": "none", "preview": response.content[:200], "trace_id": _next_trace_id()}
            ],
        }

    upload_session_id = state.get("upload_session_id")
    uploaded_document = get_uploaded_document(upload_session_id) if upload_session_id else None
    full_messages = _build_messages(state)
    has_web_results = _successful_tool_used(state, "search_web")
    has_fetch_results = _successful_tool_used(state, "fetch_url")
    if has_web_results:
        full_messages.append(SystemMessage(content=(
            "You already have usable web search results in the prior tool output. "
            "Do not call search_web again. Produce the final answer from those results, "
            "or use retrieve_docs only if the user asks a technical documentation question."
        )))
    llm = get_llm()
    llm_with_tools = llm.bind_tools(REACT_TOOLS)
    response = llm_with_tools.invoke(full_messages)

    if response.tool_calls:
        requested_tools = {tc["name"] for tc in response.tool_calls}
        should_answer_now = False
        if (
            uploaded_document
            and "search_web" in requested_tools
            and "retrieve_docs" not in requested_tools
            and not _current_web_question(state["question"])
            and not _successful_tool_used(state, "retrieve_docs")
        ):
            response = AIMessage(
                content="",
                additional_kwargs={"tool_calls": [{
                    "id": f"uploaded_retrieve_{_next_trace_id()}",
                    "type": "function",
                    "function": {
                        "name": "retrieve_docs",
                        "arguments": json.dumps({"query": state["question"]}),
                    },
                }]},
            )
            requested_tools = {"retrieve_docs"}
        if has_fetch_results:
            should_answer_now = True
        elif has_web_results and "search_web" in requested_tools:
            should_answer_now = True
        elif has_web_results and "retrieve_docs" in requested_tools and not _technical_docs_question(state["question"]):
            should_answer_now = True
        if should_answer_now:
            response = llm.invoke(full_messages + [SystemMessage(content=(
                "Answer now using the existing tool results. Do not call any more tools. "
                "Use <answer> tags for the final answer."
            ))])

    new_trace = list(state.get("trace", []))
    trace_id = _next_trace_id()

    new_trace.append({
        "step": "llm_response",
        "trace_id": trace_id,
        "content": _llm_response_log(response),
    })

    if response.tool_calls:
        for tc in response.tool_calls:
            new_trace.append({
                "step": "agent_decided",
                "trace_id": trace_id,
                "tool": tc["name"],
                "args": tc["args"],
            })
    else:
        new_trace.append({
            "step": "agent_generating",
            "trace_id": trace_id,
            "tool": "none",
            "preview": response.content[:300] if response.content else "",
            "result": response.content or "",
        })

    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
        "trace": new_trace,
    }


def execute_tool(state: GraphState) -> dict:
    """Execute a single tool based on the last AI message's tool_calls."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {"messages": []}

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    args = tool_call["args"]

    # Reuse the trace_id from the matching agent_decided entry for pairing
    trace = list(state.get("trace", []))
    call_trace_id = 0
    for entry in reversed(trace):
        if entry.get("step") == "agent_decided" and entry.get("tool") == tool_name:
            call_trace_id = entry.get("trace_id", 0)
            break

    if tool_name == "search_web":
        result = search_web.invoke(args)
    elif tool_name == "fetch_url":
        result = fetch_url.invoke(args)
    elif tool_name == "retrieve_docs":
        docs = retrieve_documents(
            args["query"],
            top_k=settings.top_k,
            upload_session_id=state.get("upload_session_id"),
        )
        if not docs:
            result = "No relevant documents found in the knowledge base."
        else:
            result = format_retrieved_documents(docs)
    elif tool_name == "run_python":
        result = run_python.invoke(args)
    else:
        result = f"Unknown tool: {tool_name}"

    trace.append({
        "step": "tool_result",
        "trace_id": call_trace_id,
        "tool": tool_name,
        "preview": str(result)[:300],
        "result": str(result),
    })

    return {
        "messages": [ToolMessage(content=str(result), tool_call_id=tool_call["id"])],
        "trace": trace,
    }


def router(state: GraphState) -> str:
    """Route after agent: tool call → execute_tool, plain answer → end."""
    last_message = state["messages"][-1]
    iteration_count = state.get("iteration_count", 0)

    if iteration_count >= settings.max_tool_iterations:
        return "end"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tool"
    return "end"


def extract_generation(state: GraphState) -> dict:
    """Extract the final generation from messages. Populated at graph END."""
    generation = ""
    sources = state.get("sources", [])

    # Prefer the last AI message without tool_calls
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            if not msg.tool_calls:
                generation = msg.content
                break

    if not generation:
        generation = "\n".join(
            msg.content for msg in state.get("messages", [])
            if isinstance(msg, AIMessage) and msg.content
        )

    return {"generation": generation, "sources": sources}


def build_graph() -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("extract", extract_generation)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        router,
        {
            "execute_tool": "execute_tool",
            "end": "extract",
        },
    )
    workflow.add_edge("execute_tool", "agent")
    workflow.add_edge("extract", END)

    return workflow.compile()


graph = build_graph()
