"""
Lore Registry tools — the deterministic/semantic pair the report calls for:
lookup_entity (deterministic, exact-match, checked first) and search_lore
(semantic, Stage 0's hybrid pipeline, checked when nothing names a specific
entity). Read order for lookup_entity: this campaign's own live NPC/Location/
Item records first (the freshest truth), then the canon Lore Registry (for an
entity not yet instantiated in this campaign), matching the source-of-truth
mapping in design.md.
"""

from langchain_core.tools import tool

from backend.stores.campaign_store import CampaignStore
from backend.stores.graph_store import RelationGraphStore
from backend.stores.lore_store import LoreEntity, LoreStore
from backend.stores.rules_store import RulesStore
from backend.tools._helpers import find_item_anywhere, find_location, find_npc


def _dm_only_label(spoiler_tier: str, text: str) -> str:
    if spoiler_tier == "dm_only" and text:
        return f"[DM-ONLY — do not reveal directly] {text}"
    return text


def _format_lore_entity(entity: LoreEntity) -> str:
    profile = entity.rolled_up_profile
    lines = [f"=== {entity.canonical_name} ({entity.entity_type}, from '{entity.book_slug}') ==="]
    if entity.aliases:
        lines.append(f"Aliases: {', '.join(entity.aliases)}")
    for key, value in profile.items():
        if not value or key == "spoiler_tier":
            continue
        text = ", ".join(value) if isinstance(value, list) else str(value)
        lines.append(_dm_only_label(entity.spoiler_tier, f"{key}: {text}"))
    if entity.source_chunk_ids:
        lines.append(f"Source chunk_ids: {', '.join(entity.source_chunk_ids)}")
    return "\n".join(lines)


def make_tools(
    campaign_id: str,
    store: CampaignStore,
    lore_store: LoreStore,
    rules_store: RulesStore,
    books_in_play: list[str],
    graph_store: RelationGraphStore | None = None,
) -> list:

    @tool
    async def lookup_entity(name: str) -> str:
        """Deterministic canonical lookup for a named NPC, location, or item —
        call this FIRST whenever a query names a specific individual/place/
        object, before search_lore/search_rules. Checks this campaign's own
        live records first (exact name or alias), then this adventure's (or
        any core rulebook's) precomputed canon Lore Registry if no live
        record exists yet. DM-only tagged fields are labeled inline
        '[DM-ONLY — do not reveal directly]' — never state them outright in
        narration unless the party has actually discovered them (see
        reveal_npc_knowledge/reveal_hidden_element). Monster stat blocks are
        NOT handled here — create_monster looks those up directly, since
        they're mechanics-grounding data, not narrative lore."""
        campaign = await store.load(campaign_id)

        npc = find_npc(campaign, name)
        if npc is None:
            npc = next((n for n in campaign.npcs if name.lower() in [a.lower() for a in n.aliases]), None)
        if npc:
            lines = [f"=== {npc.name} (live NPC record) ==="]
            if npc.aliases:
                lines.append(f"Aliases: {', '.join(npc.aliases)}")
            lines.append(f"Race/occupation: {npc.race} {npc.occupation}".strip())
            lines.append(f"Attitude: {npc.attitude.value}  Alive: {npc.is_alive}  Location: {npc.location or 'unknown'}")
            if npc.motivations:
                lines.append("Motivations: " + "; ".join(npc.motivations))
            for secret in npc.secrets:
                lines.append(_dm_only_label("dm_only" if npc.spoiler_tier == "dm_only" else "public", f"Secret: {secret}"))
            return "\n".join(lines)

        loc = find_location(campaign, name)
        if loc is None:
            loc = next((l for l in campaign.locations if name.lower() in [a.lower() for a in l.aliases]), None)
        if loc:
            lines = [f"=== {loc.name} (live location record) ==="]
            if loc.aliases:
                lines.append(f"Aliases: {', '.join(loc.aliases)}")
            if loc.description:
                lines.append(loc.description)
            if loc.points_of_interest:
                lines.append("Points of interest: " + "; ".join(loc.points_of_interest))
            for hidden in loc.hidden_elements:
                lines.append(_dm_only_label("dm_only" if loc.spoiler_tier == "dm_only" else "public", f"Hidden: {hidden}"))
            return "\n".join(lines)

        item_hit = find_item_anywhere(campaign, name)
        if item_hit:
            item, holder = item_hit
            lines = [f"=== {item.name} (live item record, in {holder}) ==="]
            if item.aliases:
                lines.append(f"Aliases: {', '.join(item.aliases)}")
            lines.append(f"Type: {item.item_type}  Magical: {item.magical}  Rarity: {item.rarity or 'mundane'}")
            if item.description:
                lines.append(_dm_only_label(item.spoiler_tier, item.description))
            return "\n".join(lines)

        # No live record — fall back to the canon Lore Registry. LoreStore
        # applies the same "core is always included" scope RulesStore.search()
        # uses, so this checks core rulebook entities regardless of
        # books_in_play, plus this campaign's adventure books.
        for entity_type in ("npc", "location", "item"):
            entity = await lore_store.find_by_name_or_alias(books_in_play, name, entity_type=entity_type)
            if entity:
                return _format_lore_entity(entity)

        return f"No entity named '{name}' found — not a live campaign record, and not in the canon Lore Registry."

    @tool
    async def search_lore(query: str, entity_type: str | None = None) -> str:
        """Semantic search over indexed lore (the hybrid dense+BM25+rerank
        pipeline also used by search_rules). entity_type optionally narrows
        which kind of canon entity to also check for an alias match
        ('npc'/'location'/'item') — leave unset to check all three. If the
        query text matches a known alias, that entity's full canon profile +
        source chunk_ids are included alongside the semantic hits,
        guaranteeing coverage rather than hoping top-k similarity catches
        it. Every passage is tagged with its chunk_id — cite it (not just
        book/section) whenever you relay a fact from here into your
        response, so it can be verified. If retrieval is thin even after a
        reformulated re-try, this says so explicitly rather than padding the
        answer — treat that as a signal to abstain, not to guess."""
        if not rules_store.is_ready():
            return "Rulebook index is not ready. Run build_index.py first, then restart the app."

        from backend.rag.grading import grade_sufficiency, reformulate_query

        chunks = rules_store.search(query, books_in_play=books_in_play)
        sufficient = grade_sufficiency(query, chunks)
        if not sufficient:
            # Bounded to exactly one retry — matches this app's universal
            # no-unbounded-retry discipline (see dm_agent.py's correction_count
            # /tool_error_count/lore_guardrail_count budgets).
            query2 = reformulate_query(query)
            reretrieved = rules_store.search(query2, books_in_play=books_in_play, wide_k=50)
            if reretrieved:
                chunks = reretrieved
                sufficient = grade_sufficiency(query2, chunks)

        parts = []
        if chunks:
            parts.append("\n\n---\n\n".join(
                f"[{c.book} — {c.section} | chunk_id: {c.chunk_id}]\n{c.content}" for c in chunks
            ))

        kinds = [entity_type] if entity_type else ["npc", "location", "item"]
        for kind in kinds:
            entity = await lore_store.find_by_name_or_alias(books_in_play, query, entity_type=kind)
            if entity:
                parts.insert(0, f"[Canon entity match]\n{_format_lore_entity(entity)}")

        if not parts:
            return f"No relevant lore found for '{query}'. Say so plainly if asked — do not invent an answer."

        result = "\n\n---\n\n".join(parts)
        if not sufficient:
            result += "\n\n[Note: retrieval may be incomplete for this query — consider saying so rather than filling gaps with invention.]"
        return result

    tools = [lookup_entity, search_lore]

    if graph_store is not None:
        @tool
        async def get_related_entities(name: str, relation_filter: str | None = None, max_hops: int = 2) -> str:
            """Deterministic multi-hop traversal over this campaign's
            relationship graph (NPC<->location, item<->NPC, faction ties,
            etc.) — answers "how is X connected to Y" / "who's affiliated
            with this place" with no LLM call for the traversal itself.
            relation_filter (e.g. "located in", "owns") only narrows DIRECT
            (1-hop) relationships — multi-hop paths are shown as "indirect"
            since a single relation label can't describe a whole chain.
            max_hops caps how far out to search (default 2)."""
            graph = await graph_store.load_networkx(campaign_id)
            name_lower = name.lower()
            start = next(
                (n for n, data in graph.nodes(data=True) if data.get("name", "").lower() == name_lower), None,
            )
            if start is None:
                return f"'{name}' has no recorded relationships in this campaign yet."

            import networkx as nx
            lengths = nx.single_source_shortest_path_length(graph, start, cutoff=max_hops)
            lines = [f"=== Entities related to {name} (within {max_hops} hop(s)) ==="]
            for node_id, hops in sorted(lengths.items(), key=lambda kv: kv[1]):
                if node_id == start:
                    continue
                data = graph.nodes[node_id]
                if hops == 1:
                    edge_data = graph.get_edge_data(start, node_id) or {}
                    relation = edge_data.get("relation", "")
                    if relation_filter and relation.lower() != relation_filter.lower():
                        continue
                    label = relation or "related to"
                else:
                    label = "indirect"
                lines.append(f"  [{hops} hop(s)] {data.get('name', '?')} ({data.get('type', '?')}) — {label}")

            if len(lines) == 1:
                return f"'{name}' has no recorded relationships in this campaign yet."
            return "\n".join(lines)

        tools.append(get_related_entities)

    return tools
