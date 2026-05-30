# LangGraph Overview

LangGraph is a framework for building stateful agent workflows on top of language models. It is useful when an application needs more control than a single prompt or a simple chain can provide: loops, branching, tool execution, human approval, persistence, streaming, retries, and multi-step state management.

LangGraph applications are modeled as graphs. Each graph has a shared state object, nodes that read and update that state, and edges that decide which node runs next. This makes LangGraph a good fit for ReAct agents, retrieval-augmented generation pipelines, multi-agent systems, human-in-the-loop review flows, and long-running assistant workflows.

## When to Use LangGraph

Use LangGraph when the application needs explicit control over execution. Common examples include:

- An agent that repeatedly decides whether to call a tool or answer.
- A RAG system that retrieves, grades, rewrites the query, and retries retrieval when results are weak.
- A support assistant that pauses for human approval before sending an email or updating a ticket.
- A coding assistant with separate planner, executor, reviewer, and test-runner steps.
- A multi-agent workflow where specialized agents hand work to each other.
- A workflow that must resume after interruption or recover from process restarts.

LangGraph is usually more appropriate than a plain chain when the control flow is dynamic, cyclic, stateful, or needs durable checkpoints.

## Core Mental Model

The core concepts are:

- **State**: a typed data structure shared across the graph.
- **Nodes**: Python functions or callables that receive state and return updates.
- **Edges**: transitions that decide which node runs next.
- **Reducers**: merge rules that define how state updates are combined.
- **Checkpointers**: persistence layers that save graph state between steps.
- **Threads**: conversation or workflow identifiers used with persistence.

A graph starts at `START`, runs nodes according to edges, and ends at `END`.

## StateGraph

`StateGraph` is the main graph builder. You define the state schema, add nodes, connect edges, compile the graph, and then invoke or stream it.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    question: str
    answer: str
    steps: list[str]

def draft_answer(state: AgentState):
    return {
        "answer": f"Draft answer for: {state['question']}",
        "steps": ["drafted answer"],
    }

builder = StateGraph(AgentState)
builder.add_node("draft_answer", draft_answer)
builder.add_edge(START, "draft_answer")
builder.add_edge("draft_answer", END)

graph = builder.compile()
result = graph.invoke({"question": "What is LangGraph?", "answer": "", "steps": []})
```

## State Schema

The state schema describes the keys that flow through the graph. A `TypedDict` is common for Python projects because it documents the shape of state and gives type checkers useful information.

```python
from typing import TypedDict

class RAGState(TypedDict):
    question: str
    documents: list[str]
    generation: str
    retry_count: int
```

Each node receives the current state and returns only the fields it wants to update. LangGraph merges those updates into the state.

## Reducers and Append-Only Fields

By default, a returned state update replaces the previous value for that key. Reducers customize this behavior. They are important for append-only fields such as chat messages, audit logs, retrieved sources, or intermediate reasoning steps.

For message-based agents, LangGraph commonly uses the `add_messages` reducer so new messages are appended rather than replacing the message history.

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
```

Use reducers when multiple nodes may contribute to the same state key or when state needs to accumulate over time.

## Nodes

Nodes are the units of work in a graph. A node can call an LLM, retrieve documents, execute a tool, transform state, validate output, or route to a human review step.

```python
def retrieve(state: RAGState):
    docs = vectorstore.similarity_search(state["question"], k=4)
    return {"documents": [doc.page_content for doc in docs]}

def generate(state: RAGState):
    context = "\n\n".join(state["documents"])
    answer = llm.invoke(f"Answer using this context:\n{context}\nQuestion: {state['question']}")
    return {"generation": answer.content}
```

Good nodes are small and named after their responsibility: `retrieve`, `grade_documents`, `rewrite_query`, `call_agent`, `execute_tool`, `human_review`, or `summarize`.

## Edges

Edges define the execution order. A normal edge always moves from one node to another.

```python
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
```

Normal edges are best when the workflow is deterministic.

## Conditional Edges

Conditional edges make the graph dynamic. A routing function reads state and returns the next node name.

```python
def route_after_retrieval(state: RAGState):
    if not state["documents"] and state["retry_count"] < 2:
        return "rewrite_query"
    return "generate"

builder.add_conditional_edges(
    "retrieve",
    route_after_retrieval,
    {
        "rewrite_query": "rewrite_query",
        "generate": "generate",
    },
)
```

Conditional edges are how LangGraph implements loops, retries, agent tool decisions, routing by intent, and branching workflows.

## ReAct Agents and Tool Calling

A ReAct-style LangGraph agent usually has two main nodes:

- **Agent node**: sends messages to the LLM and receives either a final answer or tool calls.
- **Tool node**: executes the requested tool and appends the result back to state.

The graph loops from agent to tool and back to agent until the model returns a final answer.

```text
START -> agent -> tools -> agent -> END
```

Important implementation details:

- Keep tool results in the message history so the model can reason over them.
- Add a max-iteration guard to prevent infinite loops.
- Give tools clear descriptions so the model chooses the right tool.
- Add routing rules that prevent repeated failing tool calls.
- Log tool input and output for debugging.

## Retrieval-Augmented Generation with LangGraph

LangGraph is well suited for RAG because retrieval workflows often need control flow. A robust RAG graph might include:

1. Classify whether the question needs retrieval, web search, or direct answer.
2. Retrieve documents from a vector store.
3. Grade retrieved documents for relevance.
4. Rewrite the query if retrieved documents are weak.
5. Retrieve again with the improved query.
6. Generate an answer with citations.
7. Optionally validate whether the answer is grounded in the retrieved context.

This pattern is stronger than a one-shot retrieval chain because it can recover when the initial search misses the right context.

## Persistence and Checkpointing

Persistence lets a graph save state after each step. This enables resuming interrupted workflows, maintaining conversation memory, and supporting human-in-the-loop pauses.

Checkpointers store graph state by thread. A thread is usually a conversation id, user session id, or workflow id.

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-123"}}
graph.invoke({"messages": []}, config=config)
```

For production, use a durable checkpointer rather than only in-memory storage. Persistence is essential when users expect conversations or workflows to continue across requests.

## Human-in-the-Loop Workflows

LangGraph can pause before sensitive actions. A graph might draft an email, pause for human approval, and then continue only after the human accepts or edits the action.

Human-in-the-loop is useful for:

- Sending emails or messages.
- Updating databases or tickets.
- Running destructive operations.
- Approving agent plans.
- Reviewing generated code or legal/medical content.

The graph state should include the proposed action, the approval status, and any human comments.

## Streaming

LangGraph supports streaming intermediate graph events and final model tokens. Streaming is important for UX because users can see tool decisions, retrieved context, and answer generation as they happen.

Common streaming surfaces include:

- Tokens from the final LLM response.
- Node-level updates after each graph step.
- Tool call and tool result events.
- Debug traces for agent reasoning and routing.

For FastAPI or web apps, streaming graph events through Server-Sent Events is a practical pattern.

## Memory

LangGraph applications can use short-term and long-term memory. Short-term memory usually lives in the graph state or checkpointed messages for a thread. Long-term memory is often stored in an external database or vector store and retrieved when relevant.

Design memory carefully:

- Keep recent messages for conversation continuity.
- Summarize long histories to control token usage.
- Store durable user facts only with clear consent.
- Retrieve long-term memories only when relevant to the current task.

## Multi-Agent Workflows

LangGraph can coordinate multiple agents by making each agent a node or subgraph. For example:

- Planner agent creates a plan.
- Research agent gathers context.
- Writer agent drafts output.
- Reviewer agent critiques the draft.
- Executor agent applies changes.

Use multi-agent graphs when separate roles improve reliability. Avoid adding agents when a single well-designed graph node is enough.

## Error Handling and Guardrails

Production LangGraph apps should include explicit guardrails:

- Limit max tool iterations.
- Detect repeated failed tool calls.
- Validate structured output before using it.
- Catch tool exceptions and return useful tool error messages.
- Add fallbacks for unavailable external services.
- Trace every node input, node output, and routing decision.

For web search tools, guard against rate limits and repeated searches. For retrieval tools, return a clear no-results message when no documents match.

## Testing LangGraph Applications

Test each layer separately:

- Unit test node functions with small state dictionaries.
- Unit test routing functions for each branch.
- Mock LLM responses to test graph loops deterministically.
- Test tool execution with controlled tool results.
- Integration test the compiled graph with representative questions.
- Regression test streaming event order if the UI depends on it.

Good tests make agent behavior less mysterious and reduce regressions when prompts or tools change.

## Deployment Notes

When deploying LangGraph behind an API:

- Keep graph construction importable and fast.
- Avoid blocking the web event loop with long synchronous model calls.
- Use background threads or async APIs for streaming.
- Add request-level timeouts.
- Keep user/session identifiers out of global state unless intentionally scoped.
- Use durable persistence if conversations must survive server restarts.

## Common Questions

### How is LangGraph different from a chain?

A chain is usually linear. LangGraph supports cycles, conditional routing, checkpointing, multi-step state, and human approval. Use a chain for simple deterministic flows; use LangGraph for agent workflows that need control.

### What is the state in LangGraph?

State is the shared data object passed between nodes. Nodes read state and return updates. The schema is often a `TypedDict`.

### What are nodes and edges?

Nodes do work. Edges decide what runs next. Conditional edges inspect state and choose a route.

### How do I prevent infinite agent loops?

Track an iteration count in state, set a max iteration limit, detect repeated failed tool calls, and route to `END` with a useful answer when the limit is reached.

### How should an agent choose tools?

Give each tool a specific description, add system instructions about when to use each tool, include session context such as uploaded documents, and add graph-level guardrails for known failure modes.

## Practical ReAct Graph Skeleton

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration_count: int

def agent(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "iteration_count": state["iteration_count"] + 1}

def execute_tool(state: AgentState):
    tool_call = state["messages"][-1].tool_calls[0]
    result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
    return {"messages": [ToolMessage(content=str(result), tool_call_id=tool_call["id"])]}

def route(state: AgentState):
    last = state["messages"][-1]
    if state["iteration_count"] >= 10:
        return "end"
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"

builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tools", execute_tool)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")
graph = builder.compile()
```

This skeleton is the foundation for many production agent loops.
