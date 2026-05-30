from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

_project_root = Path(__file__).parent.parent.parent
_env_file = _project_root / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_env_file),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Agentic RAG Portfolio"
    debug: bool = False
    cors_origins: list[str] = ["*"]

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "llama3"
    mock_llm: bool = False

    embedding_provider: str = "huggingface"
    embedding_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_model: Optional[str] = None

    use_local_vectorstore: bool = True
    faiss_index_path: str = ".faiss_index"

    max_retries: int = 2
    top_k: int = 3

    fast_mode: bool = False
    enable_grading: bool = True
    stream_chunk_delay: float = 0.025
    stream_chunk_size: int = 4
    llm_max_tokens: int = 4096
    llm_timeout: int = 30

    max_tool_iterations: int = 10
    max_upload_bytes: int = 2_000_000
    max_upload_session_id_length: int = 128
    trace_text_limit: int = 1200
    trace_result_limit: int = 2000


settings = Settings()
