from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://dnd_dm:dnd_dm@localhost:5432/dnd_dm"
    # No longer used by the live app as of the vLLM-metal embeddings
    # migration (vllm-migration-plan.md §7.7) — Ollama serves nothing in the
    # runtime path anymore (chat and embeddings both moved to vLLM-metal).
    # Still used by offline scripts (extract_entities.py, clean_source.py,
    # add_headers.py) via ollama_chat() — out of scope for both migrations,
    # and by the manual break-glass fallback (see vllm_base_url below).
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
    # Briefly tried the 3-bit quant instead of 4-bit (2026-07-13) to ease
    # real memory pressure on this 32GB Mac — tool_choice="required"
    # compliance held up (13/14 = 92.9%, matching 4-bit), but a live spot
    # check of actual contextualization output (the task that matters most
    # for quality, not tested by the tool-calling battery) caught a real
    # hallucination: a generic PHB combat-rules chunk got contextualized as
    # being about "the Stonehill Inn" and "Phandalin" (Lost Mine of
    # Phandelver locations with zero connection to that passage) — almost
    # certainly the model echoing the contextualizer's own few-shot example
    # instead of grounding in the actual content. Reverted to 4-bit rather
    # than risk more of that silently landing in the corpus; every
    # contextualization done under 3-bit was reset (contextualized=false)
    # and will be redone under 4-bit, not just skipped.
    mechanics_model: str = "mlx-community/Qwen3-30B-A3B-4bit"
    # Second vLLM-metal server (see vllm-migration-plan.md §7.7), served via
    # `vllm serve ... --convert embed` — a separate process/port from chat,
    # since vLLM serves one model per process. Ollama no longer serves
    # embeddings in the runtime path either, as of this migration (was
    # nomic-embed-text — a BERT-family architecture that can't run on
    # vllm-metal's MLX backend at all, confirmed live; this Qwen3-based
    # embedding model is the verified-live replacement).
    vllm_embed_base_url: str = "http://localhost:8101/v1"
    embed_model: str = "mlx-community/Qwen3-Embedding-0.6B-8bit"
    # Ollama keep_alive for ollama_chat()-built clients (offline scripts,
    # see ollama_base_url above — no longer used by any live-app client) — a
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
