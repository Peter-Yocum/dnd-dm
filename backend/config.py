from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://dnd_dm:dnd_dm@localhost:5432/dnd_dm"
    ollama_base_url: str = "http://localhost:11434"
    chroma_persist_dir: str = "data/chroma_db"
    # vLLM-metal chat server (see vllm-migration-plan.md) — reached from
    # backend/llm.py's vllm_chat() factory. host.docker.internal in the app
    # container, same reachability pattern as ollama_base_url, since
    # vllm-metal runs natively on the host Mac (not in Docker), per
    # docker-compose.yml. Ollama itself no longer serves chat in the normal
    # runtime path as of this migration — kept installed only for the
    # manual break-glass fallback documented in the migration plan.
    vllm_base_url: str = "http://localhost:8100/v1"
    # Both the mechanics and narrator nodes use this model. Was
    # "gemma4:26b-mlx" (via Ollama) until the vLLM-metal migration —
    # switched for real forced tool-calling (tool_choice="required"),
    # unavailable on Ollama's serving stack. See vllm-migration-plan.md.
    mechanics_model: str = "mlx-community/Qwen3-30B-A3B-4bit"
    embed_model: str = "nomic-embed-text"
    # Ollama keep_alive for every client built by backend/llm.py — a
    # duration string ("5m", "45m", "-1" for forever); None sends nothing
    # and leaves the Ollama server's own idle-timeout eviction in charge.
    # Deliberately None today: keep_alive=-1 was tried 2026-07-10 on both
    # the chat and embedding models chasing the embed<->chat swap freeze
    # (design.md's Evolution section). It genuinely stopped the swap, but a
    # never-reloaded model's KV cache only ever grows (confirmed live: 18GB
    # resident right after pinning, 26GB hours later, back to 17GB after an
    # `ollama stop` + fresh reload), eating headroom other native processes
    # need — a segfault in a concurrent ingest run traced back to exactly
    # this. A future bounded experiment ("30m"/"45m") is now a one-line
    # change HERE (or an OLLAMA_KEEP_ALIVE env var) instead of a hunt
    # across every construction site.
    ollama_keep_alive: str | None = None

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
