import sys
import io
import threading
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

_vectorstore_lock = threading.Lock()
_local_vectorstore_cache = None
_llm_cache = None
_uploaded_documents: dict[str, dict] = {}
_session_vectorstores: dict[str, FAISS] = {}
_uploaded_lock = threading.Lock()

_project_root = Path(__file__).parent.parent.parent
ALLOWED_UPLOAD_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".docx"}


def get_llm(max_tokens: int = None) -> ChatOpenAI:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=0,
            max_tokens=max_tokens or settings.llm_max_tokens,
            request_timeout=settings.llm_timeout,
        )
    return _llm_cache


def get_embeddings():
    if settings.embedding_provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        return OpenAIEmbeddings(
            base_url=settings.embedding_base_url or settings.llm_base_url,
            api_key=settings.embedding_api_key or settings.llm_api_key,
            model=settings.embedding_model or "nomic-embed-text",
        )


def call_llm(prompt: str) -> str:
    if settings.mock_llm:
        return _mock_llm_response(prompt)
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content


async def acall_llm(prompt: str) -> str:
    if settings.mock_llm:
        return _mock_llm_response(prompt)
    llm = get_llm()
    response = await llm.ainvoke(prompt)
    return response.content


def _mock_llm_response(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "classif" in prompt_lower and "intent" in prompt_lower:
        if any(w in prompt_lower for w in ["greeting", "hello", "hi ", "math", "2+2"]):
            return "direct"
        elif any(w in prompt_lower for w in ["news", "latest", "current", "new"]):
            return "search"
        return "retrieve"
    elif "grader" in prompt_lower or "relevance" in prompt_lower:
        return "sufficient"
    elif "rewriter" in prompt_lower or "rewrite" in prompt_lower:
        return "What is LangGraph state management?"
    return "LangGraph is a library for building stateful, multi-actor applications with LLMs."


def get_vectorstore():
    global _local_vectorstore_cache

    if _local_vectorstore_cache is not None:
        return _local_vectorstore_cache

    with _vectorstore_lock:
        if _local_vectorstore_cache is not None:
            return _local_vectorstore_cache
        index_path = _project_root / "tech-doc-rag" / settings.faiss_index_path
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. Run ingestion first."
            )
        embeddings = get_embeddings()
        _local_vectorstore_cache = FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )
    return _local_vectorstore_cache


# Eager-load the vectorstore at import time so it fails fast on startup
# and never contends with concurrent streaming threads.
get_vectorstore()


def _document_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def _read_uploaded_text(filename: str, content_type: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    normalized_content_type = content_type.lower()

    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError("unsupported upload format; upload Markdown, text, PDF, or DOCX")

    if suffix in {".md", ".markdown", ".txt"} or normalized_content_type.startswith("text/"):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("uploaded text document must be UTF-8 encoded") from exc

    if suffix == ".pdf" or normalized_content_type == "application/pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF uploads require the pypdf dependency") from exc

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if suffix == ".docx" or normalized_content_type.endswith("wordprocessingml.document"):
        try:
            from docx import Document as DocxDocument
        except ImportError as exc:
            raise ValueError("DOCX uploads require the python-docx dependency") from exc

        docx = DocxDocument(io.BytesIO(data))
        return "\n".join(paragraph.text for paragraph in docx.paragraphs).strip()

    raise ValueError("unsupported upload format; upload Markdown, text, PDF, or DOCX")


def ingest_uploaded_document(session_id: str, filename: str, content_type: str, data: bytes) -> dict:
    normalized_session_id = session_id.strip()
    if not normalized_session_id:
        raise ValueError("upload session id is required")
    if not data:
        raise ValueError("uploaded document is empty")
    if len(data) > settings.max_upload_bytes:
        size_mb = settings.max_upload_bytes / 1_000_000
        raise ValueError(f"uploaded document is too large; limit is {size_mb:.1f} MB")

    safe_filename = Path(filename or "uploaded-document").name
    content = _read_uploaded_text(safe_filename, content_type, data)
    if not content.strip():
        raise ValueError("uploaded document contains no extractable text")

    metadata = {
        "source": "uploaded",
        "path": safe_filename,
        "session_id": normalized_session_id,
        "uploaded": True,
    }
    document = Document(page_content=content, metadata=metadata)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents([document])
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    # Merge the base knowledge base into the session vectorstore.
    # Load a fresh copy from disk so the cached base (get_vectorstore()) is never touched.
    embeddings = get_embeddings()
    index_path = _project_root / "tech-doc-rag" / settings.faiss_index_path
    try:
        base_copy = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
        vectorstore.merge_from(base_copy)
    except Exception:
        pass

    row = {
        "id": f"uploaded:{normalized_session_id}:{safe_filename}",
        "title": _document_title(content, Path(safe_filename).stem or "Uploaded document"),
        "source": "uploaded",
        "path": safe_filename,
        "content": content,
        "size_bytes": len(data),
        "line_count": len(content.splitlines()),
        "session_id": normalized_session_id,
        "uploaded": True,
    }

    with _uploaded_lock:
        _uploaded_documents[normalized_session_id] = row
        _session_vectorstores[normalized_session_id] = vectorstore

    return row


def reset_uploaded_document(session_id: str) -> None:
    normalized_session_id = session_id.strip()
    with _uploaded_lock:
        _uploaded_documents.pop(normalized_session_id, None)
        _session_vectorstores.pop(normalized_session_id, None)


def get_uploaded_document(session_id: str) -> Optional[dict]:
    normalized_session_id = session_id.strip()
    with _uploaded_lock:
        document = _uploaded_documents.get(normalized_session_id)
        return dict(document) if document else None


def _search_with_scores(vectorstore, query: str, top_k: int) -> list[tuple[Document, float]]:
    if hasattr(vectorstore, "similarity_search_with_score"):
        try:
            scored = vectorstore.similarity_search_with_score(query, k=top_k)
            if isinstance(scored, list) and scored:
                return scored
        except Exception:
            pass
    return [(doc, float(index)) for index, doc in enumerate(vectorstore.similarity_search(query, k=top_k))]


def retrieve_documents(query: str, top_k: int = 3, upload_session_id: Optional[str] = None) -> list[Document]:
    if upload_session_id:
        with _uploaded_lock:
            session_vectorstore = _session_vectorstores.get(upload_session_id.strip())
        if session_vectorstore is not None:
            try:
                return [doc for doc, _ in _search_with_scores(session_vectorstore, query, top_k)[:top_k]]
            except Exception:
                return []

    try:
        return [doc for doc, _ in _search_with_scores(get_vectorstore(), query, top_k)[:top_k]]
    except FileNotFoundError:
        return []
    except Exception:
        return []


def document_source_label(doc: Document) -> str:
    metadata = doc.metadata or {}
    source = str(metadata.get("source") or "documentation")
    path = str(metadata.get("path") or metadata.get("file_path") or "").strip()
    raw_source = str(metadata.get("source") or "").strip()

    if source == "uploaded":
        return f"uploaded/{path or 'document'}"
    if path:
        return path
    if raw_source:
        try:
            data_root = (_project_root / "tech-doc-rag" / "data").resolve()
            source_path = Path(raw_source).resolve()
            return source_path.relative_to(data_root).as_posix()
        except (OSError, ValueError):
            return Path(raw_source).name or raw_source
    return source


def format_retrieved_documents(docs: list[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        source_label = document_source_label(doc)
        formatted.append(
            f"[Document {i} | Source: {source_label}]\n"
            f"Use this exact readable citation inline next to claims from this document: [{source_label}]\n"
            "Do not save citations for a final Sources or References section.\n"
            "Do not use numeric line references.\n"
            f"{content}"
        )
    return "\n\n---\n\n".join(formatted)


# ---------------------------------------------------------------------------
# ReAct tools
# ---------------------------------------------------------------------------


import urllib.request
import urllib.parse
import time as _time
from html.parser import HTMLParser

_SEARCH_LAST_CALL: float = 0.0


class _DDGResultParser(HTMLParser):
    """Parse DuckDuckGo HTML search results into {title, href, snippet} dicts."""

    def __init__(self):
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._cur: dict[str, str] = {}
        self._in_title = False
        self._in_snippet = False
        self._ad_depth = 0

    def _cls(self, attrs):
        return dict(attrs).get("class", "")

    def handle_starttag(self, tag, attrs):
        cls = self._cls(attrs)
        if tag == "div" and "result--ad" in cls:
            self._ad_depth += 1
            return
        if self._ad_depth > 0:
            return
        if tag == "a" and "result__a" in cls:
            self._cur = {"href": dict(attrs).get("href", ""), "title": "", "snippet": ""}
            self._in_title = True
            self._in_snippet = False
        elif tag == "a" and "result__snippet" in cls:
            self._in_snippet = True
            self._in_title = False
            if "href" not in self._cur:
                self._cur = {"href": "", "title": "", "snippet": ""}

    def handle_data(self, data):
        if self._ad_depth > 0 or "href" not in self._cur:
            return
        if self._in_title:
            self._cur["title"] += data
        elif self._in_snippet:
            self._cur["snippet"] += data

    def handle_endtag(self, tag):
        if self._ad_depth > 0:
            if tag == "div":
                self._ad_depth -= 1
            return
        if "href" not in self._cur:
            return
        if tag == "a" and self._in_title:
            self._in_title = False
        elif tag == "a" and self._in_snippet:
            self._in_snippet = False
            self.results.append(self._cur)
            self._cur = {}


def _ddg_fetch(query: str, timeout: int = 30) -> str:
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp.read().decode("utf-8", errors="replace")


def _parse_ddg_results(html: str) -> list[dict[str, str]]:
    parser = _DDGResultParser()
    parser.feed(html)
    return parser.results


def _extract_url(href_raw: str) -> str:
    if "uddg=" in href_raw:
        parsed = urllib.parse.urlparse(href_raw)
        qs = urllib.parse.parse_qs(parsed.query)
        return qs.get("uddg", [href_raw])[0]
    return href_raw


def _search_ddg(query: str) -> str:
    """Execute a single DDG search with backoff, return formatted string."""
    global _SEARCH_LAST_CALL
    elapsed = _time.time() - _SEARCH_LAST_CALL
    if elapsed < 3.0:
        _time.sleep(3.0 - elapsed)

    try:
        html = _ddg_fetch(query, timeout=30)
    except Exception:
        _time.sleep(3.0)
        try:
            html = _ddg_fetch(query, timeout=60)
        except Exception:
            return "__RATE_LIMITED__"
    _SEARCH_LAST_CALL = _time.time()

    # Check for DDG challenge/captcha page
    if "result__a" not in html:
        return "__RATE_LIMITED__"

    results = _parse_ddg_results(html)
    if not results:
        return "__NO_RESULTS__"

    seen = set()
    formatted = []
    for r in results:
        title = r.get("title", "").strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        href = _extract_url(r.get("href", ""))
        snippet = r.get("snippet", "").strip()
        formatted.append(f"Title: {title}\nURL: {href}\nContent: {snippet}")
        if len(formatted) >= 5:
            break

    return "\n\n---\n\n".join(formatted)


@tool
def search_web(query: str) -> str:
    """Search the web for current information. Use this for news, recent events, and topics needing up-to-date information."""
    try:
        result = _search_ddg(query)
        if result == "__RATE_LIMITED__":
            _time.sleep(5.0)
            result = _search_ddg(query)
        if result == "__RATE_LIMITED__":
            return "Web search is temporarily rate-limited by DuckDuckGo. Please try a different approach or wait before searching again."
        if result == "__NO_RESULTS__":
            return "No search results found."
        return result
    except Exception as e:
        return f"Search error: {e}"


class _TextExtractParser(HTMLParser):
    """Extract readable title/body text from fetched HTML pages."""

    def __init__(self):
        super().__init__()
        self.title = ""
        self.parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_data(self, data):
        text = " ".join(data.split())
        if not text or self._skip_depth > 0:
            return
        if self._in_title:
            self.title += text
        elif len(text) > 2:
            self.parts.append(text)

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False


@tool
def fetch_url(url: str) -> str:
    """Fetch a web page URL and extract readable text. Use after search_web returns relevant links."""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return "Fetch error: only http/https URLs are supported."
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        resp = urllib.request.urlopen(req, timeout=15)
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.read(1_000_000)
        if "text/html" not in content_type and "text/plain" not in content_type:
            return f"Fetch error: unsupported content type {content_type or 'unknown'}."
        text = raw.decode("utf-8", errors="replace")
        if "text/plain" in content_type:
            body = " ".join(text.split())[:12000]
            return f"URL: {url}\nContent: {body}"
        parser = _TextExtractParser()
        parser.feed(text)
        body = " ".join(parser.parts)[:12000]
        title = parser.title.strip() or url
        return f"Title: {title}\nURL: {url}\nContent: {body}"
    except Exception as e:
        return f"Fetch error: {e}"


@tool
def retrieve_docs(query: str) -> str:
    """Search the documentation knowledge base and the active temporary uploaded document, if any."""
    try:
        docs = retrieve_documents(query, top_k=settings.top_k)
    except Exception as e:
        return f"Error retrieving documents: {e}"

    if not docs:
        return "No relevant documents found in the knowledge base."

    return format_retrieved_documents(docs)


import ast as _ast


@tool
def run_python(code: str) -> str:
    """Execute Python code in a sandboxed environment. Use for computations, testing code snippets, or data analysis."""
    output = []
    exception = []

    def target():
        try:
            import builtins

            forbidden = {"eval", "exec", "compile", "open", "input", "__import__"}
            safe_builtins = {
                k: v
                for k, v in vars(builtins).items()
                if k not in forbidden and not k.startswith("_")
            }
            namespace = {"__builtins__": safe_builtins}

            # Auto-print the last expression if it's not already printed
            try:
                tree = _ast.parse(code)
                if tree.body and isinstance(tree.body[-1], _ast.Expr):
                    last_expr = tree.body[-1]
                    wrapper = _ast.fix_missing_locations(
                        _ast.Module(
                            body=tree.body[:-1]
                            + [_ast.Expr(_ast.Call(func=_ast.Name(id="print", ctx=_ast.Load()), args=[last_expr.value], keywords=[]))],
                            type_ignores=[],
                        )
                    )
                    compiled = compile(wrapper, "<run_python>", "exec")
                else:
                    compiled = compile(code, "<run_python>", "exec")
            except SyntaxError:
                compiled = compile(code, "<run_python>", "exec")

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                exec(compiled, namespace)
                out = sys.stdout.getvalue()
                output.append(out if out.strip() else "Code executed successfully (no output)")
            except Exception as e:
                output.append(f"Error: {type(e).__name__}: {e}")
            finally:
                sys.stdout = old_stdout
        except Exception as e:
            exception.append(str(e))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=10)

    if exception:
        return f"Execution error: {exception[0]}"
    if output:
        return output[0]
    return "Error: Code execution timed out (10s limit)"


# ReAct tool list (for binding to the LLM)
REACT_TOOLS = [search_web, fetch_url, retrieve_docs, run_python]
