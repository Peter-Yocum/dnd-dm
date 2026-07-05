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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
