"""
LoreStore — the canon Lore Registry: book-scoped, campaign-agnostic entity
profiles (NPC/Location/Item/Monster) precomputed offline by
scripts/extract_entities.py --write-postgres. Never mutated by live play —
see backend/models.py's LoreLinked docstring and design.md's source-of-truth
mapping for how this relates to a campaign's own live NPC/Location/Item rows.

SQLAlchemy Core, mirrors CampaignStore's connection pattern (AsyncEngine,
async with self._engine.begin()/.connect()).
"""

from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import delete, insert, or_, select
from sqlalchemy.ext.asyncio import AsyncEngine

from backend.stores import tables as t


def _book_scope(book_slugs: list[str]):
    """Mirrors RulesStore._build_where's "core is always included" filter:
    matches any core-sourced entity regardless of book_slugs, plus any
    adventure entity whose book_slug is in books_in_play. Returns None (no
    filter — matches everything) only if book_slugs is falsy AND the caller
    wants an unscoped search; callers here always pass at least this filter."""
    if not book_slugs:
        return t.lore_entities.c.source_type == "core"
    return or_(
        t.lore_entities.c.source_type == "core",
        t.lore_entities.c.book_slug.in_(book_slugs),
    )


class LoreEntity(BaseModel):
    id: str
    book_slug: str
    source_type: str
    entity_type: str
    canonical_name: str
    rolled_up_profile: dict
    source_chunk_ids: list[str]
    spoiler_tier: str
    aliases: list[str]


class LoreStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def upsert_entity(
        self,
        book_slug: str,
        entity_type: str,
        canonical_name: str,
        profile: dict,
        aliases: list[str],
        source_chunk_ids: list[str] | None = None,
        spoiler_tier: str = "public",
        source_type: str = "adventure",
    ) -> str:
        """Insert-or-update keyed on (book_slug, entity_type, canonical_name)
        — safe to call repeatedly (e.g. a re-extracted book, or a resumed
        run) without creating duplicate rows."""
        source_chunk_ids = source_chunk_ids or []
        async with self._engine.begin() as conn:
            existing = (await conn.execute(
                select(t.lore_entities.c.id).where(
                    t.lore_entities.c.book_slug == book_slug,
                    t.lore_entities.c.entity_type == entity_type,
                    t.lore_entities.c.canonical_name == canonical_name,
                )
            )).scalar_one_or_none()

            entity_id = existing or uuid4().hex
            values = {
                "book_slug": book_slug,
                "source_type": source_type,
                "entity_type": entity_type,
                "canonical_name": canonical_name,
                "rolled_up_profile": profile,
                "source_chunk_ids": source_chunk_ids,
                "spoiler_tier": spoiler_tier,
            }
            if existing:
                await conn.execute(
                    t.lore_entities.update()
                    .where(t.lore_entities.c.id == entity_id)
                    .values(**values)
                )
                await conn.execute(delete(t.lore_entity_aliases).where(
                    t.lore_entity_aliases.c.lore_entity_id == entity_id
                ))
            else:
                await conn.execute(insert(t.lore_entities).values(id=entity_id, **values))

            if aliases:
                await conn.execute(insert(t.lore_entity_aliases).values([
                    {"id": uuid4().hex, "lore_entity_id": entity_id, "alias": alias}
                    for alias in dict.fromkeys(aliases)  # dedupe, preserve order
                ]))

        return entity_id

    async def find_by_name_or_alias(self, book_slugs: list[str], name: str, entity_type: str | None = None) -> LoreEntity | None:
        """Exact match (case-insensitive) against canonical_name or any
        alias. Core-sourced entities always match regardless of book_slugs
        (same "core is always included" convention as RulesStore.search());
        adventure-sourced entities only match if their book_slug is in
        book_slugs."""
        conditions = [_book_scope(book_slugs)]
        if entity_type:
            conditions.append(t.lore_entities.c.entity_type == entity_type)

        async with self._engine.connect() as conn:
            direct = await conn.execute(
                select(t.lore_entities).where(
                    t.lore_entities.c.canonical_name.ilike(name), *conditions,
                )
            )
            row = direct.mappings().first()
            if row is None:
                via_alias = await conn.execute(
                    select(t.lore_entities)
                    .select_from(t.lore_entities.join(
                        t.lore_entity_aliases,
                        t.lore_entity_aliases.c.lore_entity_id == t.lore_entities.c.id,
                    ))
                    .where(t.lore_entity_aliases.c.alias.ilike(name), *conditions)
                )
                row = via_alias.mappings().first()
            if row is None:
                return None

            aliases = (await conn.execute(
                select(t.lore_entity_aliases.c.alias).where(
                    t.lore_entity_aliases.c.lore_entity_id == row["id"]
                )
            )).scalars().all()

        return LoreEntity(
            id=row["id"], book_slug=row["book_slug"], source_type=row["source_type"],
            entity_type=row["entity_type"], canonical_name=row["canonical_name"],
            rolled_up_profile=row["rolled_up_profile"], source_chunk_ids=list(row["source_chunk_ids"] or []),
            spoiler_tier=row["spoiler_tier"], aliases=list(aliases),
        )

    async def all_for_book(self, book_slug: str, entity_type: str) -> list[LoreEntity]:
        """All entities of one type for one specific book (not core-scoped —
        an exact book_slug match only). Used by Stage 1.5's canon relation
        seeding, which needs every item's found_at/owned_by profile field
        for the book just extracted, not a name-lookup."""
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                select(t.lore_entities).where(
                    t.lore_entities.c.book_slug == book_slug,
                    t.lore_entities.c.entity_type == entity_type,
                )
            )).mappings().all()
            entities = []
            for row in rows:
                aliases = (await conn.execute(
                    select(t.lore_entity_aliases.c.alias).where(
                        t.lore_entity_aliases.c.lore_entity_id == row["id"]
                    )
                )).scalars().all()
                entities.append(LoreEntity(
                    id=row["id"], book_slug=row["book_slug"], source_type=row["source_type"],
                    entity_type=row["entity_type"], canonical_name=row["canonical_name"],
                    rolled_up_profile=row["rolled_up_profile"], source_chunk_ids=list(row["source_chunk_ids"] or []),
                    spoiler_tier=row["spoiler_tier"], aliases=list(aliases),
                ))
        return entities

    async def find_candidates(self, book_slugs: list[str], entity_type: str) -> list[str]:
        """All canonical names + aliases for fuzzy-blocking (dedup checks in
        create_npc/create_location/add_item_to_character). Same core-always-
        included scope as find_by_name_or_alias."""
        conditions = [t.lore_entities.c.entity_type == entity_type, _book_scope(book_slugs)]

        async with self._engine.connect() as conn:
            names = (await conn.execute(
                select(t.lore_entities.c.canonical_name).where(*conditions)
            )).scalars().all()
            aliases = (await conn.execute(
                select(t.lore_entity_aliases.c.alias)
                .select_from(t.lore_entity_aliases.join(
                    t.lore_entities, t.lore_entities.c.id == t.lore_entity_aliases.c.lore_entity_id,
                ))
                .where(*conditions)
            )).scalars().all()
        return list(names) + list(aliases)
