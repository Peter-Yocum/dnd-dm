"""
RelationGraphStore — a LightRAG-style, set-merging, campaign-scoped
relationship graph (NPC<->faction, NPC<->location, item<->location, ...),
updated incrementally as new sessions/entities are added. No Neo4j, no full
rebuilds: entity_relations' UniqueConstraint on
(campaign_id, source_id, target_id, relation) IS the set-merging mechanism —
re-adding the same fact is a harmless upsert no-op.

Protocol earns its keep here (unlike a single-implementation store like
LoreStore) because the user explicitly named two plausible backends up
front: a Postgres edge table (shipped now) vs. a NetworkX-graph-pickled-to-
disk file (a clean future swap if a much larger campaign's edge count ever
makes Postgres joins the bottleneck — no caller changes needed if that swap
happens, since callers only see this Protocol).
"""

from typing import Protocol
from uuid import uuid4

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from backend.stores import tables as t


class RelationGraphStore(Protocol):
    async def add_edge(
        self, campaign_id: str, source_type: str, source_id: str, source_name: str,
        target_type: str, target_id: str, target_name: str, relation: str,
        description: str = "", source_chunk_ids: list[str] | None = None,
    ) -> None: ...

    async def load_networkx(self, campaign_id: str) -> "networkx.Graph": ...


class PostgresRelationGraphStore:
    """The one implementation shipped now. load_networkx rebuilds fresh from
    Postgres rows on every call — campaign-scale graphs are small (tens to
    low-hundreds of edges, per the source report's own cited '50-100 named
    NPCs per 20-30 session campaign' scale), so this is cheap and avoids a
    second persisted graph format entirely."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def add_edge(
        self, campaign_id: str, source_type: str, source_id: str, source_name: str,
        target_type: str, target_id: str, target_name: str, relation: str,
        description: str = "", source_chunk_ids: list[str] | None = None,
    ) -> None:
        stmt = pg_insert(t.entity_relations).values(
            id=uuid4().hex, campaign_id=campaign_id,
            source_type=source_type, source_id=source_id, source_name=source_name,
            target_type=target_type, target_id=target_id, target_name=target_name,
            relation=relation, description=description, source_chunk_ids=source_chunk_ids or [],
        )
        # ON CONFLICT DO NOTHING on the (campaign_id, source_id, target_id,
        # relation) unique constraint — re-adding the same fact is a no-op,
        # not an error. This IS the incremental set-merging mechanism.
        stmt = stmt.on_conflict_do_nothing(constraint="uq_entity_relations_edge")
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def load_networkx(self, campaign_id: str):
        import networkx as nx
        from sqlalchemy import select

        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                select(t.entity_relations).where(t.entity_relations.c.campaign_id == campaign_id)
            )).mappings().all()

        graph = nx.Graph()
        for row in rows:
            graph.add_node(row["source_id"], name=row["source_name"], type=row["source_type"])
            graph.add_node(row["target_id"], name=row["target_name"], type=row["target_type"])
            graph.add_edge(
                row["source_id"], row["target_id"],
                relation=row["relation"], description=row["description"],
            )
        return graph
