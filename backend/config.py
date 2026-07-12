from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://dnd_dm:dnd_dm@localhost:5432/dnd_dm"
    ollama_base_url: str = "http://localhost:11434"
    chroma_persist_dir: str = "data/chroma_db"
    # Both the mechanics and narrator nodes use this model — benchmarked
    # against a smaller dedicated narrator model (gemma4:12b-mlx) and found
    # to be both faster at raw generation (~37.5 vs ~25.3 tok/s) and to
    # incur no swap cost either way (both fit resident simultaneously), so
    # a second model bought nothing. See design.md tech stack notes.
    mechanics_model: str = "gemma4:26b-mlx"
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
