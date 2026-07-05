# Module 7 — The DM Bridge (Capstone)

You've built every component in the sandbox. This module is the design sketch for assembling them into your Dungeon Master, plus the two swaps that turn a toy into something that survives closing your laptop. It's intentionally a scaffold with real decisions left to you — you're past tutorial mode now.

## What you already know how to do

| DM need | Sandbox skill | Module |
|---|---|---|
| Run a local model | `ChatOllama` | 1 |
| Persona + reusable prompts | prompt templates / LCEL | 2 |
| Roll dice, do math | `@tool` functions | 3 |
| Decide *when* to act | `create_agent` loop | 3 / 5b |
| Remember the session | checkpointer + `thread_id` | 4 |
| Look up rules from books | RAG (retrieve + ground) | 5 |
| Let the agent choose to look up | RAG as a tool (agentic) | 5b |
| Ingest scanned/handwritten books | vision OCR → markdown | 6 |

The DM is just these wired together. No new concepts — only assembly and two production upgrades.

## Recommended project layout

Split the one-file scripts into a small project. Separation matters because the prep steps (OCR, indexing) run rarely, while the DM loop runs every session.

```
dnd-dm/
├── .venv/
├── docs/
│   ├── raw/             # scanned/handwritten PDFs (input to OCR)
│   └── source/          # OCR'd + HAND-CHECKED .md (the real source of truth)
├── ocr_ingest.py        # Module 6 tool — run when you add scanned books
├── build_index.py       # docs/source/*.md  ->  persistent vector store
├── tools.py             # roll_dice, search_rules
├── dm.py                # the agent + chat loop
├── chroma_db/           # persisted embeddings (gitignore this)
└── dm.sqlite            # persisted conversation memory (gitignore this)
```

Runtime data flow: `dm.py` loads the already-built `chroma_db/` and `dm.sqlite`. It never OCRs or re-embeds — those happen ahead of time via the two prep scripts.

## Swap #1 — Persistent memory (`SqliteSaver`)

`InMemorySaver` forgot everything on exit. For a campaign that resumes week to week, swap in SQLite — same checkpointer interface, but it writes to disk.

```bash
pip install -U langgraph-checkpoint-sqlite
```

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# check_same_thread=False so the REPL can reuse the connection across turns
conn = sqlite3.connect("dm.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)
# ...pass checkpointer=checkpointer to create_agent, exactly like Module 4.
```

Now a `thread_id` like `"curse-of-strahd"` reloads that campaign's entire history next week. Different campaigns = different `thread_id`s, fully isolated (the Module 4 lesson, now durable).

## Swap #2 — Persistent rules index (`Chroma`)

`InMemoryVectorStore` re-embedded every run. Chroma saves vectors to disk so you embed each book once.

```bash
pip install -U langchain-chroma
```

`build_index.py` (run only when books change):

```python
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter,
)

def split_markdown(md_text, max_chars=2000, overlap=200):
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "subsection")],
        strip_headers=False,
    ).split_text(md_text)
    return RecursiveCharacterTextSplitter(
        chunk_size=max_chars, chunk_overlap=overlap
    ).split_documents(sections)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = Chroma(
    collection_name="rules",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# TODO A: for each .md in docs/source/, read it, split_markdown(...) it,
#   and store.add_documents(chunks). Tag metadata with the book name so
#   citations can say WHICH book + section. (Chroma auto-persists to disk.)
for md_file in Path("docs/source").glob("*.md"):
    ...
print("Index built to ./chroma_db")
```

`dm.py` then *loads* that store without re-adding (constructing `Chroma` with the same `persist_directory` and `collection_name` reopens it).

## `tools.py` — the DM's hands

```python
import random, re
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Load the prebuilt index once, at import.
_store = Chroma(
    collection_name="rules",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory="./chroma_db",
)

@tool
def search_rules(query: str) -> str:
    """Look up D&D rules, spells, monsters, or items in the rulebooks. Use this
    for ANY question about how a rule works or what a game element does. Returns
    relevant excerpts with their book and section.
    """
    hits = _store.similarity_search(query, k=4)
    return "\n\n".join(
        f"[{d.metadata.get('book','?')} — {d.metadata.get('section','?')}]\n{d.page_content}"
        for d in hits
    )

@tool
def roll_dice(notation: str) -> str:
    """Roll dice in standard notation like '1d20', '2d6+3', '4d6'. Use whenever
    a roll is needed (attacks, saves, damage, checks). Returns the individual
    dice and the total.
    """
    # TODO B: parse NdM(+/-K), roll N M-sided dice, sum, apply modifier.
    #   Return something like "1d20+5: [17] +5 = 22". Keep it deterministic-free
    #   (real randomness) and validate bad input instead of crashing.
    ...
```

Note `roll_dice` is the **exact** shape of the Module 3 `scale_recipe` tool — small, well-described, single job. And `search_rules` is `search_recipes` with the nouns changed. You already wrote both; here you're just renaming.

## `dm.py` — assembly + chat loop

```python
import sqlite3
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from tools import search_rules, roll_dice

model = ChatOllama(model="qwen2.5:14b", temperature=0.7)  # creative for narration
conn = sqlite3.connect("dm.sqlite", check_same_thread=False)

DM_SYSTEM = (
    "You are a Dungeon Master running a D&D 5e game. Narrate vividly but concisely.\n"
    "- For ANY rules/spell/monster question, call search_rules and ground your "
    "answer in it; cite the book and section. Never invent a rule.\n"
    "- When the situation calls for a roll, call roll_dice; never make up results.\n"
    "- If the rulebooks don't cover something, say so and clearly label any "
    "ruling as YOUR improvisation, not official text.\n"
    "- Track the party, location, and current scene from the conversation so far."
)

# TODO C: create the agent with both tools, the DM_SYSTEM prompt, and the
#   SqliteSaver(conn) checkpointer.
agent = ...

def main():
    campaign = input("Campaign name (thread id): ").strip() or "default-campaign"
    config = {"configurable": {"thread_id": campaign}}
    print(f"\n[{campaign}] — type 'quit' to save & exit.\n")
    while True:
        player = input("You: ").strip()
        if player.lower() in {"quit", "exit"}:
            break
        # TODO D: invoke the agent with the player's message + config,
        #   then print the last message's content as "DM: ...".
        ...

if __name__ == "__main__":
    main()
```

Run prep once, then play:

```bash
python ocr_ingest.py      # only for scanned books; skip for clean digital PDFs
python build_index.py     # embed the rulebooks
python dm.py              # play — resumes the campaign by thread_id each time
```

## DM-specific design decisions (the interesting part)

These are judgment calls the sandbox didn't force; your DM does.

- **Grounding vs. improvisation.** The single most important prompt rule: make the model *label* what's from the book vs. what it's making up. You saw in 5b that a fluent model will confidently improvise (the gluten-free-soup misread). For rulings, that's a feature *if* it's labeled, a bug if it masquerades as official text.
- **Two memories.** Short-term (the checkpointer) is this session's transcript. You'll also want **long-term** facts that outlive a session — party roster, character sheets, the artifact found three sessions ago. That's the LangGraph `Store` (previewed in Module 4), or honestly, for a solo project, a plain JSON file you inject into the system prompt. Don't over-engineer it early.
- **Dice as a tool, not the model's imagination.** Always route randomness through `roll_dice`. A model "rolling" in its head isn't random and isn't auditable. The tool gives you real entropy and a visible result.
- **Model split.** Narration wants a creative temperature; rules lookup and dice want determinism. Options: one model at ~0.7 and trust the tools for the exact bits, or two `ChatOllama` instances at different temperatures. Start with one; split only if narration leaks into rulings.
- **Citations build trust.** Carrying `book`/`section` metadata through to the answer means players can verify. It's also your debugging handle when a ruling looks wrong — you can see exactly which chunk it used.

## Verify it like an engineer

When you wire it up, don't just "play" — test the seams you already know are fragile:
- Ask a rules question that IS in your books → confirm it calls `search_rules` and cites.
- Ask one that ISN'T → confirm it admits the gap and labels its improvisation.
- Trigger a roll → confirm it calls `roll_dice`, not invents a number.
- Quit and relaunch the same `thread_id` → confirm the party/scene survived (memory persisted).
- Flip `set_debug(True)` for one exchange to watch the tool-calling sequence, exactly as in 5b.

## Stretch goals (when the core works)

- Streaming output (`agent.stream`) so narration appears as it's generated.
- A simple web UI (the agent core doesn't change — only the I/O around it).
- Per-character long-term memory in the `Store`.
- An `update_party` tool so the model can record HP/inventory changes structurally.
- Swap `search_rules` for a multi-book retriever that filters by which books are "in play."

## You're off the rails now

Every piece here is something you built and debugged in the sandbox — local models, tools, the agent loop, memory, RAG, OCR, observability, and the current-vs-legacy instincts to read docs critically. The DM is assembly plus taste. Build the skeleton (`tools.py` + `dm.py` with one book indexed), get one full exchange working end to end, then grow it. When you hit something gnarly, you now know how to isolate it: turn on debug, look at what was retrieved, check what the tool actually received. That diagnostic habit is the real takeaway — more than any single API.

Go run your game.