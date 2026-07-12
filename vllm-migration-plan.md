# vLLM-Metal Migration Plan: full backend swap, real forced tool-calling everywhere

**Status:** Not started. This is a self-contained implementation plan — hand this whole
file to a fresh Claude Code session to execute it; it doesn't assume any prior
conversation context.

**Prerequisite reading (optional but useful):** `docs/VERIFICATION.md` (this repo's
manual, scenario-driven verification convention — no automated test suite, by design).

**Revision note:** an earlier draft of this plan proposed a hybrid design (Ollama stays
primary, vLLM-metal only backs the guardrail-retry path as a second always-resident
backend). That's been abandoned: this Mac has 32GB total RAM, Ollama's
`gemma4:26b-mlx` is 17GB, and vLLM-metal claimed a ~25GB Metal memory ceiling in
testing — running both resident simultaneously doesn't fit without aggressively killing
processes to free RAM on demand, which defeats the point of an always-warm fallback.
This revision does a **full backend swap** instead: retire Ollama for the app's main
model entirely, replace it everywhere with vLLM-metal, and use the headroom that frees
up to force tool-calling far more broadly than just the retry path.

---

## 1. Why this exists

`backend/agent/dm_agent.py` runs a two-node LangGraph agent (mechanics → narrator) for
the in-game DM. The mechanics node is a tool-calling loop backed by Ollama
(`settings.mechanics_model = "gemma4:26b-mlx"`, via `langchain_ollama.ChatOllama`). A
long guardrail chain in that file (`_detect_missing_followup`,
`_detect_missing_combat_roll_followup`, `_detect_missing_loot_followup`,
`_detect_missing_encounter_followup`, `_detect_stalled_non_player_turn_followup`, the
Stage-2 lore guardrails) exists because the model sometimes narrates an outcome (a hit,
damage, a loot gain) without ever making the tool call that makes it real.

**Root cause: Ollama has no real `tool_choice`/forced-tool-calling mechanism.**
Confirmed by reading the installed `langchain_ollama==1.1.0` source directly —
`ChatOllama.bind_tools(tool_choice=...)`'s own docstring says *"This parameter is
currently ignored as it is not supported by Ollama."* The underlying `ollama==0.6.2`
Python client's `chat()` call has no `tool_choice` field at all. This isn't a LangChain
integration gap — the capability doesn't exist in Ollama's serving stack.

**Real forcing exists in mainline vLLM.** `tool_choice="required"` is genuinely enforced
there via structured-output/structural-tag constraints. A spike (branch
`spike/vllm-tool-choice-forcing`, commits `4d3c8e3`..`3e5f15d`) tested this empirically
on this exact Mac against a production-equivalent model:

- Installed **vllm-metal** (official vLLM Apple Silicon plugin) into `~/.venv-vllm-metal`
  — clean install, no Rosetta, native arm64.
- Sourced **`mlx-community/gemma-4-26b-a4b-it-nvfp4`** (~7GB) — architecture, parameter
  count (26.2B), and quantization (`nvfp4`) all match the locally-tagged
  `gemma4:26b-mlx` Ollama model. Very likely the same upstream checkpoint.
- Served it with `--enable-auto-tool-choice --tool-call-parser gemma4 --max-model-len 8192`.
- Ran `scripts/spikes/vllm_tool_choice/bench_tool_choice.py` — real 5e-shaped prompts,
  real tool schemas from `backend/tools`, bound with `tool_choice="required"`.
- **Result: 40/40 = 100% compliance.** Never once returned plain content instead of a
  tool call.
- **Throughput: 22.9 tok/s steady-state** (one cold-start outlier excluded — first
  request took ~76s to compile/warm up, every request after was 22-24 tok/s) vs. the
  ~37.5 tok/s Ollama baseline noted in `backend/config.py`'s comment — **~39% slower.**
  This cost now applies to **every** call across the app, not just retries — see §3.

Full raw output: `scripts/spikes/vllm_tool_choice/results-gemma4-26b-nvfp4.txt`.

**A second, independent reason to do this migration**, found while scoping it: this
codebase has a **documented, recurring class of production hangs** from Ollama swapping
between the embedding model (`nomic-embed-text`) and the chat model
(`gemma4:26b-mlx`) within a single process. See `backend/agent/world_prep.py` (line 50-62,
`_gather_seed_context`'s docstring) for one confirmed live incident ("the MLX runner
stuck reporting 'Stopping...' indefinitely") and its point-fix (removing an embedding
call from that one code path); `backend/agent/dm_agent.py` around line 2003 for a second,
separately point-fixed instance in session summarization; and `backend/rag/rules.py`'s
`search_rules` tool, which sets `use_reranker=False` specifically to avoid triggering
this swap on the most-called live-gameplay lookup. **A full swap removes the root cause
everywhere at once, permanently**, instead of requiring another ad-hoc workaround the
next time this bug surfaces in a new code path — because after this migration, Ollama
only ever serves one model (`nomic-embed-text`), so there is no swap to trigger.

**Known open risk, carried forward, not resolved by the spike:** Gemma 4 sits on
vllm-metal's own documented support matrix at the **experimental** tier (Gemma 3 is full
support). It worked cleanly in the spike, but a future vllm-metal release could change
that in either direction. `nvfp4` isn't in vllm-metal's documented quantization list
(`AWQ`, GGUF `Q8_0`/`Q4_0`) either — it loaded and ran correctly here, but that's "worked
once," not a documented guarantee. Full swap makes this risk more consequential than the
hybrid design would have (see §6) — there's no Ollama fallback quietly still running.

---

## 2. Scope: every `settings.mechanics_model` call site moves to vLLM

`grep`-confirmed: **the entire app's non-embedding LLM usage is one model**
(`settings.mechanics_model = "gemma4:26b-mlx"`), across these construction sites:

| File | Site | Role |
|---|---|---|
| `backend/agent/dm_agent.py:134` | `_get_model()` | world-prep, party-fill, session summarization, and other single-pass agents (`create_react_agent`, no mechanics/narrator split) |
| `backend/agent/dm_agent.py:181` | `_get_mechanics_model()` | in-game mechanics node + Session Zero's `chargen_mechanics_node` |
| `backend/agent/dm_agent.py:199` | `_get_narrator_model()` | in-game narrator node + Session Zero's `chargen_narrator_node` |
| `backend/rag/contextualizer.py:41` | contextualization pass (chunk augmentation during indexing) | |
| `backend/rag/grading.py:27,58` | `grade_sufficiency` / `reformulate_query` (CRAG-style retrieval grading) | |
| `backend/rag/reranker.py:87` | `LLMJudgeReranker` | |

**Embeddings stay on Ollama, unchanged** — `nomic-embed-text` (274MB, per `ollama list`)
via `OllamaEmbeddings` in `backend/stores/rules_store.py:67` and
`backend/stores/history_store.py:55,129`. Trivial RAM footprint, and per §1's second
reason, isolating it as the *only* thing Ollama serves is itself a fix, not just a thing
left alone.

**Migration = replace every `ChatOllama(model=settings.mechanics_model, ...)`
construction above with an equivalent `langchain_openai.ChatOpenAI` pointed at
vLLM-metal's OpenAI-compatible endpoint.** Same model identity throughout (one served
checkpoint), same call shapes (temperature/reasoning params carry over conceptually),
different client library and base URL.

---

## 3. Recommended design

### 3.1 One persistent vLLM-metal server, replacing Ollama for chat entirely

Single long-lived vLLM-metal process serving `mlx-community/gemma-4-26b-a4b-it-nvfp4`
(or whichever final checkpoint — see §7's open question on quantization choice),
OpenAI-compatible, reached from every call site in §2's table. Ollama keeps running,
but only for `nomic-embed-text`.

### 3.2 Force tool-calling on *every* mechanics call, not just retries — via a `conclude_turn` tool

This is the part that changes now that RAM isn't forcing a retry-only scope.

**The blocker in the original (hybrid) plan:** `tool_choice="required"` forces a tool
call on every invocation it's bound to. But `mechanics_node`'s loop currently
terminates precisely when the model responds *without* a tool call — that response's
plain text becomes `notes`, the resolution report handed to the narrator
(`dm_agent.py` line 1082: `notes = _extract_text(response.content)`, and the
`if response.tool_calls:` branch at line 1067 is what routes to `"tools"` instead of
falling through to build that report). Force every call and the model could never
produce that terminal plain-text response — the loop would never end.

**The fix: give the model a way to "finish" that's *itself* a tool call.** Add a new
tool, `conclude_turn(resolution_notes: str)`, to the mechanics tool set. Its "execution"
is trivial (just carries the notes through — no game-state mutation, no store write).
With this tool always available, `tool_choice="required"` can be bound on **every**
mechanics call: the model always has a real option (a genuine game-mechanics tool call)
*or* a way to signal "I'm done" (`conclude_turn`) — it's never boxed into narrating with
no tool call available, because that option no longer needs to exist.

**`mechanics_node` control-flow change** (`dm_agent.py`, around lines 1034-1082 today):
- Bind `tool_choice="required"` on the model used for every `mechanics_model.ainvoke(...)`
  call (not just the retry path).
- After the call, check: is `conclude_turn` among `response.tool_calls`? (Should be the
  only call in that response when present — the model is signaling "done," not doing one
  more action and also concluding.)
  - **Yes** → extract `resolution_notes` from that tool call's `args`. This *replaces*
    `notes = _extract_text(response.content)` as the source of the resolution report.
    Skip the `"tools"` node entirely for this call (no real state to execute) and fall
    through into the existing guardrail chain / narrator handoff logic completely
    unchanged from here — the guardrail chain doesn't care where `notes` came from,
    only what it says.
  - **No** (a real game-mechanics tool call, or several) → `goto="tools"`, exactly as
    today. Unchanged.

**What this eliminates structurally, at the source, not just statistically:** the "zero
tool calls at all during an active encounter" branch of `_detect_missing_followup`
(`dm_agent.py` line 518, the `if not called:` branch inside it) and the whole reason
`_detect_missing_combat_roll_followup` (line 585) had to exist as a *not-gated-to-combat*
backstop — a response with literally no tool call becomes impossible, because
`tool_choice="required"` no longer has a plain-text escape hatch. These guardrail
*functions* don't need to be deleted (see §5 — deleting working safety code on a plan
document's say-so, before it's proven live, is exactly the kind of overreach to avoid),
but their **triggering condition should never actually fire again** once this ships. If
it does, that's a real vLLM/parser bug worth investigating immediately, not expected
noise — log it loudly (see §8).

**What does NOT go away:** `conclude_turn` can still be called with fabricated
`resolution_notes` — the model narrating a hit/loot gain inside `resolution_notes`
*text* without having made the real `resolve_attack`/`add_item_to_character`/etc. call
first. `tool_choice="required"` guarantees *a* tool call happened, never that it was the
*semantically correct* one, or that the model didn't just move the old "narrate without
calling anything" failure mode one level down into `conclude_turn`'s argument instead of
raw response content. **The existing content-based checks inside
`_detect_missing_combat_roll_followup`, `_detect_missing_loot_followup`,
`_detect_missing_encounter_followup`, and the Stage-2 lore guardrails still matter and
still need to run** — just point them at `conclude_turn`'s `resolution_notes` argument
instead of `_extract_text(response.content)`. Mechanical adaptation, not a redesign of
their detection logic (the regexes, the "was the turn actually advanced" check inside
`_detect_missing_followup`, etc. are unchanged).

### 3.3 Retry path: keep it, make it stronger with `conclude_turn` excluded

The guardrail-retry mechanism (`_retry_bound_model()`, `dm_agent.py` line 1014,
`_RETRY_TOOLS_*` constants) stays, now pointed at the same vLLM server as everything
else (no second backend — see the revision note at the top). One change: **exclude
`conclude_turn` from every `_RETRY_TOOLS_*` narrowed set.** A retry fires because a
guardrail already detected a real problem (a combat roll narrated with no backing call,
a stalled non-player turn, etc.) — on that retry, the model should be forced into a
*real* resolution tool, not given the option to `conclude_turn` its way past the retry
with another round of unbacked narration. This closes the one gap forcing alone doesn't:
a model could otherwise satisfy `tool_choice="required"` on a retry by just calling
`conclude_turn` again with slightly different fabricated text.

### 3.4 Narrator, world-prep/party-fill/summarization, RAG components

No `tool_choice` concerns — narrator has no tools bound at all, and `_get_model()`'s
callers use `create_react_agent`'s own tool-loop or no tools (RAG contextualizer/grading/
reranker are plain generation/classification calls). These just need the client swap
from §2's table — `ChatOllama` → `ChatOpenAI` pointed at the vLLM endpoint, same
temperature/params. No control-flow changes.

### 3.5 Session Zero (`chargen_mechanics_node`/`chargen_narrator_node`)

Automatically inherits the backend swap for free — it calls the same
`_get_mechanics_model()`/`_get_narrator_model()` factories (`dm_agent.py` lines 181, 199).
Adopting the same `conclude_turn`-and-universal-forcing pattern there too (it has an
analogous problem — `_detect_fake_tool_call`, `_detect_invented_spells`, around line
1428 onward) is a natural, low-effort follow-up once this pattern is proven live in the
main agent, but is **not spec'd out in this plan** — keep this migration's first pass
scoped to `get_agent()`'s `mechanics_node`.

---

## 4. Resource budget

Full swap changes this math for the better vs. the abandoned hybrid design — only one
large model is ever resident, not two:

- vLLM-metal: one instance, `mlx-community/gemma-4-26b-a4b-it-nvfp4`, self-claimed
  ~25GB Metal ceiling in the spike (tunable down via `--gpu-memory-utilization`, per
  the KV-cache error message vLLM printed when this was first tested — see the spike's
  `results-gemma4-26b-nvfp4.txt` for the exact numbers that error surfaced).
- Ollama: `nomic-embed-text` only, 274MB. Trivial, always-resident is fine.
- Total on this 32GB Mac: comfortably fits without evicting anything, unlike the
  abandoned hybrid design's 17GB+25GB math.

**Still verify before shipping, because the *usage pattern* changed, not just the model
count:** every one of §2's call sites now hits the same vLLM server — world-prep/
party-fill background passes, RAG contextualization during indexing, and live gameplay
mechanics/narrator calls could all land concurrently in a way they never did when spread
across Ollama (with its own internal queuing) and a separate retry-only vLLM instance.
This app is documented elsewhere as single-user/low-throughput, so this is likely a
non-issue, but confirm vLLM's default concurrency handling (`--max-num-seqs` and
friends) doesn't serialize a live player's turn behind an unrelated background
world-prep call in a way that introduces a noticeable stall. Not a hard blocker, just
worth a real check during rollout (§8), given it wasn't a risk in the old architecture
at all.

---

## 5. What does NOT change

- The guardrail chain's *detection logic* (the regexes, the "was the turn advanced"
  check, the lore-citation/abstention/spoiler checks) — same functions, same logic,
  just re-pointed at `conclude_turn`'s argument instead of raw response content where
  applicable (§3.2). **Do not delete any guardrail function as part of this migration**,
  even ones whose trigger condition should now be structurally impossible — verify that
  live (§8) before ever removing code that's still a real safety net if the `vLLM`/
  `conclude_turn` design has a gap nobody's found yet.
- Retry budgets (`correction_count`, `lore_guardrail_count`, `stalled_turn_guardrail_count`,
  all capped at 1 retry per player turn) — unchanged. §3.3's `conclude_turn` exclusion
  makes a retry's forced call stronger, not more numerous.
- `_make_tool_node`, the `"tools"` graph node, `ToolNode` wiring, tool *execution* —
  identical regardless of which model/backend requested the call.
- The narrator node's own logic and guardrails (`_detect_fake_tool_call` equivalents
  don't apply there — it never had tools).

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **No resident fallback if vLLM-metal goes down** — this is new and more serious than the abandoned hybrid design, which always had Ollama quietly still running. A vLLM crash/hang now takes down mechanics, narrator, world-prep, RAG grading/reranking — the whole app's LLM surface. | Real process supervision with auto-restart (`launchd` `KeepAlive` or equivalent — see §7's open question) is not optional here, unlike in the hybrid plan where it was just a latency nicety. Additionally: keep a **documented, manual, not-resident "break glass" procedure** — Ollama itself stays installed and `gemma4:26b-mlx` stays pulled (just not running); if vLLM-metal is down for an extended outage, a human can flip `settings`'s base URL back to Ollama and `ollama run gemma4:26b-mlx` to restore service within a couple minutes, at zero ongoing RAM cost while unused. This is different from the abandoned "always-warm fallback" — it costs nothing until the day it's actually needed. |
| Gemma 4 is "experimental" tier in vllm-metal | Worked cleanly in the spike; re-verify after any vllm-metal version bump. Keep `bench_tool_choice.py` (already in `scripts/spikes/vllm_tool_choice/`) as a regression check to re-run post-upgrade. **More consequential now** than under the hybrid plan — this is the only backend, not a rarely-used escalation path. |
| `nvfp4` isn't a documented-supported quantization | Worked once in the spike; if it breaks on a vllm-metal upgrade, fall back to a documented format from the same `mlx-community/gemma-4` collection (`-bf16`, `-8bit`, `-4bit` variants all exist) |
| `conclude_turn` design has an unknown gap (e.g. the model calls it alongside other real tool calls in the same response, against the "should be the only call" assumption in §3.2) | Handle defensively: if `conclude_turn` appears alongside other tool calls in one response, treat it as a real-tool-calls response (route to `"tools"` as normal) and log a warning — don't silently drop the other calls or silently trust `conclude_turn`'s notes while other calls are still pending execution. Verify this case explicitly during testing (§8), don't just assume it won't happen. |
| Universal forcing surfaces a NEW failure mode: the model spamming trivial/wrong tool calls just to satisfy `tool_choice="required"` when it genuinely has nothing to do (e.g. pure roleplay, an out-of-combat question) | Watch for this specifically in early live testing — this exact scenario (a turn with no game-mechanics need) is common outside combat, and `conclude_turn` should be the model's obvious/only reasonable choice there, but confirm it behaves that way rather than, say, calling `roll_dice` pointlessly. If this happens, the fix is prompting (tell the model explicitly when `conclude_turn` alone is the right call), not backing off the forcing. |
| vLLM handling all app traffic surfaces concurrency contention not present before | §4's "still verify" paragraph — real check during rollout, not assumed away |

---

## 7. Implementation steps

Assumes §4's resource check passes. All file/line references are against `main` as of
this plan's writing — re-check before editing, they will have shifted.

### 7.1 Stand up vllm-metal as a persistent local service

- Not containerized — matches how Ollama itself is run today (per `docker-compose.yml`,
  the app reaches Ollama via `OLLAMA_BASE_URL=http://host.docker.internal:11434`, i.e.
  Ollama runs natively on the host Mac, not inside Docker). Run vllm-metal the same way.
- Pick a stable port not already in use (the spike used `8100` — Docker Desktop's proxy
  already holds `8000`; verify at deploy time with `lsof -nP -iTCP:<port> -sTCP:LISTEN`).
- **Real process supervision is required, not optional** (see §6's top risk) — a
  `launchd` plist with `KeepAlive` (Mac-native, matches "runs alongside Ollama"
  precedent) or equivalent. Document the exact chosen mechanism and command line here
  once decided:
  ```
  source ~/.venv-vllm-metal/bin/activate
  vllm serve mlx-community/gemma-4-26b-a4b-it-nvfp4 \
    --port 8100 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --max-model-len <VALUE — size against real dm_agent.py usage, see _MAX_MESSAGES=100> \
    --gpu-memory-utilization <VALUE — tune per §4>
  ```
- Add a startup health-check (`curl -s http://localhost:8100/v1/models` returning 200)
  that the app itself checks at boot, since there's no Ollama fallback to silently work
  around a not-yet-ready vLLM server anymore — fail loudly and early if it's not up,
  rather than having the first player turn discover it.

### 7.2 New dependency

Add `langchain-openai` to `requirements.txt` — not currently installed (confirmed:
`ModuleNotFoundError: No module named 'langchain_openai'` in the app's `.venv`).

**Verify before relying on it** (the whole reason this migration exists is not trusting
a library's `tool_choice` claim without checking): confirm
`langchain_openai.ChatOpenAI.bind_tools(tools, tool_choice="required")` actually
forwards `tool_choice` into the request body, the way `langchain_ollama` famously does
**not**:
```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(base_url="http://localhost:8100/v1", api_key="unused", model="mlx-community/gemma-4-26b-a4b-it-nvfp4")
bound = model.bind_tools([...], tool_choice="required")
print(bound.kwargs)  # confirm 'tool_choice' key is actually present, not silently dropped
```

### 7.3 New config (`backend/config.py`)

Replace/extend the Ollama-specific settings with a vLLM equivalent, keeping
`ollama_base_url` around since embeddings still need it:
```python
vllm_base_url: str = "http://localhost:8100/v1"  # host.docker.internal in the app
                                                    # container, per docker-compose.yml
mechanics_model: str = "mlx-community/gemma-4-26b-a4b-it-nvfp4"  # now the vLLM-served
                                                                    # model name, not an
                                                                    # Ollama tag
```
Naming is a placeholder — match this file's existing style when implementing.

### 7.4 `backend/agent/dm_agent.py` changes

- **`_get_model()` (line 134), `_get_mechanics_model()` (line 181),
  `_get_narrator_model()` (line 199)**: swap `ChatOllama(...)` for
  `ChatOpenAI(base_url=settings.vllm_base_url, api_key="unused", model=settings.mechanics_model, ...)`.
  Carry over `temperature` per-factory as today. Re-evaluate whether the
  `reasoning=False` workaround (documented at length in `_get_mechanics_model()`'s
  docstring — Gemma's `<|channel>thought...<channel|>` leaking into `.content` when
  reasoning isn't explicitly disabled) is still needed or has an equivalent on
  `ChatOpenAI`/vLLM's side — the spike's clean `tool_calls`-only responses are a good
  sign this isn't a problem via vLLM's `gemma4` parser, but explicitly verify rather
  than assume (§8).
- **New `conclude_turn` tool** — add alongside the other mechanics tools (wherever
  `get_tools()` assembles its list, `backend/tools/registry.py`), or as a standalone
  tool defined directly in `dm_agent.py` if it doesn't belong conceptually with the
  game-mechanics tools (it's not touching campaign state, just a "loop control" tool
  and no `Campaign`-mutating tool it should sit next to).
- **`mechanics_node`** (starts at line 1034): implement §3.2's control flow — bind
  `tool_choice="required"` on the model for every call, check for `conclude_turn` in
  `response.tool_calls` before the existing `if response.tool_calls: goto="tools"` check
  at line 1067, extract `resolution_notes` from it in place of
  `notes = _extract_text(response.content)` (line 1082) when present.
- **`_retry_bound_model()`** (line 1014): exclude `conclude_turn` from every
  `_RETRY_TOOLS_*` set per §3.3. Point at the same vLLM client the rest of the app uses
  (no second backend).
- **Guardrail functions taking `notes`** (`_detect_missing_combat_roll_followup`,
  `_detect_missing_loot_followup`, `_detect_missing_encounter_followup`, the Stage-2
  lore guardrails) — unchanged internally; just confirm their caller passes
  `conclude_turn`'s extracted `resolution_notes` as `notes` when that's the source,
  same as it passes `_extract_text(response.content)` today.

### 7.5 `backend/rag/contextualizer.py`, `backend/rag/grading.py`, `backend/rag/reranker.py`

Same `ChatOllama` → `ChatOpenAI` swap as §7.4's factories, no tool binding involved —
plain generation/classification calls. Straightforward.

### 7.6 `docker-compose.yml`

Add the vLLM equivalent of the existing `OLLAMA_BASE_URL: http://host.docker.internal:11434`
line — a new env var (`VLLM_BASE_URL` or whatever §7.3 settles on) pointed at
`host.docker.internal:<vllm-port>`.

---

## 8. Verification plan

Follow `docs/VERIFICATION.md`'s existing convention (no automated test suite by
design — manual, scenario-driven, check real state not narration).

1. **§4's resource check** — confirm vLLM-metal alone (tuned `--gpu-memory-utilization`/
   `--max-model-len`) fits comfortably, and do a real concurrency smoke test (a
   background world-prep-shaped call + a live mechanics-shaped call close together) —
   this app never had to worry about this under the old per-role-backend architecture.
2. **`conclude_turn` wiring** — before any live-model testing, a deterministic check
   (extend `scripts/check_retry_tool_narrowing.py`, which already validates the narrowed
   tool sets) confirming: `conclude_turn` is excluded from every `_RETRY_TOOLS_*` set,
   and is present in the full/normal tool binding.
3. **Re-run `scripts/spikes/vllm_tool_choice/bench_tool_choice.py`** against the real
   persistent service from §7.1 (not the spike's ad-hoc server) — confirm 100%
   compliance holds under the actual deployed config.
4. **`conclude_turn` edge case** (§6's risk table) — deliberately construct a scenario
   where a real tool call and `conclude_turn` could both seem plausible in one response;
   confirm the app's defensive handling (§6) behaves as designed, not just in the common
   case.
5. **Live playtest**, per `docs/VERIFICATION.md`'s method: play real sessions covering
   both combat (the guardrail chain's original motivating scenarios — see `BEHAVIOR.md`)
   and clearly non-combat turns (roleplay, questions) — confirm `conclude_turn` is used
   naturally and cleanly in the latter, not fought against or spammed with junk calls
   (§6's "universal forcing surfaces a new failure mode" risk).
6. **`reasoning=False` equivalent check** (§7.4) — confirm no reasoning/thinking-channel
   leakage into tool-call arguments or `conclude_turn`'s `resolution_notes`, across
   several real turns, not just the spike's clean benchmark run.
7. **Failure-injection test**: stop the vllm-metal service mid-session, confirm the app
   fails clearly (not silently hangs) and that the manual break-glass procedure (§6)
   actually restores service when followed.
8. **Confirm the "should never fire again" guardrail branches actually don't** — after a
   reasonable amount of live play, check logs for whether
   `_detect_missing_followup`'s "no tool calls at all" branch or
   `_detect_missing_combat_roll_followup` have fired even once post-migration. If they
   have, that's a `conclude_turn`/forcing bug to chase down, not expected noise (§3.2).

---

## 9. Open questions to resolve during implementation

- Exact `--gpu-memory-utilization` / `--max-model-len` values from §4/§7.1 — fill in
  once measured against real `dm_agent.py` usage patterns (see `_MAX_MESSAGES = 100`).
- Final process-supervision mechanism for §7.1 (`launchd` plist vs. something else) —
  now load-bearing, not optional (§6) — pick and document.
- Whether to keep the `gemma4:26b-mlx` Ollama tag pulled (for the manual break-glass
  fallback in §6) or let it lapse — recommend keeping it, the disk cost is small
  relative to the operational value of a fast manual recovery path.
- Whether `conclude_turn` belongs in `backend/tools/registry.py` alongside the real
  mechanics tools or defined separately in `dm_agent.py` — implementation-detail choice,
  make it during §7.4.
- Session Zero adopting the same `conclude_turn`/universal-forcing pattern (§3.5) —
  natural follow-up, not scoped here.
- If the "no resident fallback" single-point-of-failure risk (§6) proves too costly in
  practice, revisit whether a *lighter-weight* always-on fallback (a smaller/faster
  Ollama model, not full parity, just enough to keep the app minimally functional during
  a vLLM outage) is worth the RAM cost after all — informed by real outage frequency
  data, not speculation.
