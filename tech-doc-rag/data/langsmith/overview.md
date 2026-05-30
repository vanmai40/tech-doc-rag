# LangSmith Overview

LangSmith is an observability, testing, evaluation, and monitoring platform for applications built with LLMs. It helps developers inspect traces, debug agent/tool behavior, build datasets, run evaluations, collect feedback, and monitor production quality.

LangSmith is useful because LLM applications are often nondeterministic. A normal log line rarely explains why an agent chose a tool, which prompt was sent, which retrieved documents were used, or why a final answer changed. LangSmith records structured traces so teams can inspect the full execution path.

## When to Use LangSmith

Use LangSmith when you need to answer questions like:

- What prompt and model inputs produced this answer?
- Which tool did the agent call and what did the tool return?
- Which retrieved documents were included in the context?
- Why did latency or token usage increase?
- Which examples fail after a prompt or model change?
- Is the production application improving or regressing over time?

LangSmith is especially helpful for LangChain and LangGraph applications, but it can also trace custom LLM systems.

## Core Concepts

The most important LangSmith concepts are:

- **Trace**: the full recorded execution of an LLM application request.
- **Run**: one operation inside a trace, such as an LLM call, retriever call, tool call, chain step, or agent step.
- **Project**: a workspace that groups related traces.
- **Dataset**: a collection of examples used for testing and evaluation.
- **Example**: one input/output pair in a dataset.
- **Evaluator**: logic that scores outputs for correctness, relevance, groundedness, style, or other criteria.
- **Feedback**: human or automated signal attached to a run.

## Tracing

Tracing captures the execution tree of an LLM application. A trace can show nested calls such as router -> retriever -> LLM -> tool -> LLM -> final answer.

Typical trace data includes:

- User input.
- Prompt messages.
- Model name and provider.
- Model output.
- Token usage.
- Latency.
- Tool call arguments.
- Tool results.
- Retriever queries and returned documents.
- Errors and exceptions.
- Tags and metadata.

Tracing is the first thing to enable when debugging agent behavior because it shows what actually happened rather than what the developer expected to happen.

## Basic LangChain/LangGraph Integration

For LangChain or LangGraph apps, tracing can usually be enabled with environment variables.

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api_key"
os.environ["LANGCHAIN_PROJECT"] = "tech-doc-rag-dev"
```

After tracing is enabled, model calls, chains, tools, retrievers, and LangGraph nodes can appear in LangSmith traces.

## Projects

A LangSmith project groups traces. Use separate projects for environments and workflows, for example:

- `tech-doc-rag-dev`
- `tech-doc-rag-staging`
- `tech-doc-rag-prod`
- `agent-eval-runs`

Separating projects makes it easier to compare development experiments with production behavior.

## Runs and Run Trees

A run is a unit of execution. Runs are nested into a run tree. For an agent, a single user request might contain:

1. Root chain or graph run.
2. Agent LLM run that decides to call a tool.
3. Tool run for retrieval or web search.
4. Agent LLM run that reads the tool result.
5. Final parser or response formatting run.

The run tree is useful because it shows where time was spent, which step failed, and what intermediate data influenced the final answer.

## Metadata and Tags

Tags and metadata make traces searchable. Add them to identify user flows, experiments, environments, or feature flags.

Useful tags:

- `rag`
- `react-agent`
- `upload-doc`
- `web-search`
- `prod`
- `eval`

Useful metadata:

- `user_id`
- `session_id`
- `model`
- `prompt_version`
- `retriever_top_k`
- `document_source`
- `deployment_sha`

Avoid storing secrets or sensitive user data in metadata.

## Debugging Agents

LangSmith is valuable for debugging agents because agent failures often happen before the final answer. A trace can reveal:

- The model selected the wrong tool.
- The tool description was ambiguous.
- The retrieval query was poorly formed.
- The retriever returned irrelevant documents.
- The agent repeated a failing tool call.
- The final answer ignored tool output.
- The prompt lacked session context, such as an uploaded document.

For a LangGraph ReAct loop, inspect each agent decision and each tool result. If the model calls web search when it should retrieve local docs, check the system prompt, tool descriptions, and session context included before tool selection.

## Debugging RAG

For retrieval-augmented generation, LangSmith can help answer:

- What query was sent to the retriever?
- Which chunks were returned?
- Were the chunks relevant?
- Did the prompt include the retrieved context?
- Did the answer cite the retrieved source?
- Did a query rewrite improve retrieval?

Good RAG traces include retrieved document metadata such as source path, title, chunk id, and score. This makes it easier to identify whether the problem is ingestion, retrieval, prompting, or generation.

## Datasets

Datasets store test examples. Each example usually has inputs and optional reference outputs.

```python
from langsmith import Client

client = Client()
dataset = client.create_dataset("tech-doc-rag-questions")

client.create_example(
    inputs={"question": "How does StateGraph route between nodes?"},
    outputs={"answer": "StateGraph uses edges and conditional edges to route."},
    dataset_id=dataset.id,
)
```

Datasets are useful for regression testing prompts, retrievers, tools, and full agent workflows.

## Evaluation

Evaluations run an application over a dataset and score the outputs. Evaluators can be automated or human-reviewed.

Common evaluator types:

- **Exact match**: output must match a reference string.
- **LLM-as-judge**: an LLM scores correctness, helpfulness, or groundedness.
- **Heuristic evaluator**: custom Python checks for required citations, format, or keywords.
- **Pairwise comparison**: compare two versions of an app and choose the better output.
- **Retrieval evaluator**: checks whether retrieved chunks contain the expected answer source.

For RAG systems, evaluate both retrieval quality and final answer quality. A fluent answer is not enough if it is not grounded in the retrieved documents.

## Evaluating Agent Workflows

Agent evaluation should check more than the final answer. Useful questions include:

- Did the agent choose the right tool?
- Did it avoid unnecessary web search?
- Did it stop after enough information was gathered?
- Did it recover from tool errors?
- Did it cite retrieved sources inline?
- Did it avoid calling dangerous tools without approval?

LangSmith traces make these checks easier because tool calls and intermediate outputs are structured runs.

## Feedback

Feedback records user or reviewer judgments on a run. Feedback can be numeric, categorical, or free text.

Examples:

- Thumbs up/down from users.
- `correct`, `partially_correct`, `incorrect` labels.
- Helpfulness score from 1 to 5.
- Reviewer comment explaining a hallucination.

Feedback can later become dataset examples for evaluation.

## Prompt Management

LangSmith can be used to manage prompts and prompt versions. Prompt management helps teams track which prompt produced which trace and compare prompt changes during evaluation.

Good prompt management practices:

- Version prompts.
- Record prompt version in trace metadata.
- Evaluate prompt changes against a dataset before shipping.
- Keep production prompt changes reviewable.

## Monitoring Production Applications

In production, LangSmith can monitor:

- Request volume.
- Error rate.
- Latency.
- Token usage and cost.
- Model/provider behavior.
- User feedback trends.
- Failing examples or regressions.

Monitoring is different from debugging. Debugging answers what happened in one trace. Monitoring answers whether the whole system is healthy over time.

## Handling Sensitive Data

LLM traces may contain user inputs, retrieved documents, prompts, tool outputs, and generated answers. Treat trace data carefully.

Recommended practices:

- Avoid logging secrets.
- Redact sensitive fields before tracing when needed.
- Use environment-specific projects.
- Limit access to production traces.
- Be careful with personally identifiable information.

## Practical Debugging Workflow

When an LLM app gives a bad answer:

1. Open the trace for the request.
2. Check the root input and final output.
3. Inspect the agent's tool decision.
4. Inspect retriever or web search tool inputs.
5. Inspect returned documents or tool results.
6. Check the final prompt sent to the model.
7. Decide whether the bug is routing, retrieval, prompt wording, tool output, model behavior, or missing data.
8. Add a dataset example that reproduces the failure.
9. Run an evaluation before and after the fix.

## LangSmith with LangGraph

LangSmith pairs well with LangGraph because graph applications have many intermediate steps. A trace can show each graph node, routing decision, tool call, and final output.

For a LangGraph app, useful trace metadata includes:

- graph name
- node name
- thread id
- tool name
- retrieval source
- upload session id
- retry count
- route decision

This metadata makes it easier to debug why a graph followed one branch instead of another.

## Common Questions

### What is LangSmith used for?

LangSmith is used to trace, debug, evaluate, and monitor LLM applications. It helps developers inspect what happened inside chains, agents, tools, retrievers, and graph workflows.

### Is LangSmith only for LangChain?

No. LangSmith integrates naturally with LangChain and LangGraph, but it can also be used with custom LLM applications through tracing APIs.

### What is the difference between tracing and evaluation?

Tracing records what happened for one run. Evaluation runs many examples and scores outputs to measure quality systematically.

### How does LangSmith help with RAG?

LangSmith can show the retrieval query, retrieved chunks, metadata, prompt context, model output, citations, and latency. This helps identify whether a RAG failure came from ingestion, retrieval, prompting, or generation.

### How should I start using LangSmith?

Enable tracing in development, inspect real traces, create a small dataset of important questions, add evaluators for correctness and citation quality, then monitor production traces and feedback.

## Minimal Evaluation Example

```python
from langsmith import Client

client = Client()

dataset = client.create_dataset("rag-smoke-tests")
client.create_example(
    inputs={"question": "What are LangGraph conditional edges?"},
    outputs={"must_include": "route"},
    dataset_id=dataset.id,
)

def contains_required_word(run, example):
    output = run.outputs.get("answer", "").lower()
    required = example.outputs["must_include"].lower()
    return {"key": "contains_required_word", "score": int(required in output)}
```

This kind of evaluator is simple but useful for smoke tests. More advanced evaluations can use semantic comparison or LLM judges.
