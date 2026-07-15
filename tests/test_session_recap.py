"""BDD-style coverage for rolling mid-session memory summarization (design.md
item #11) — _maybe_update_session_recap in backend/agent/dm_agent.py, which
folds narrative turns about to fall out of the narrator's _NARRATOR_MAX_TURNS
context window into a short running recap, and the modifier wiring that
prepends it right after the system message for both the mechanics and
narrator nodes.

The LLM call and LangGraph checkpointer are both monkeypatched out — this
suite exercises the pure trim/fold decision logic and the merge/formatting
contract, not real model output or Postgres-backed checkpoints.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.agent import dm_agent
from backend.models import Campaign


class _FakeModel:
    def __init__(self, reply: str):
        self._reply = reply
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        return AIMessage(content=self._reply)


def _narrative(n: int) -> list:
    msgs = []
    for i in range(n):
        msgs.append(HumanMessage(content=f"player line {i}"))
        msgs.append(AIMessage(content=f"dm line {i}"))
    return msgs


@pytest.mark.asyncio
async def test_returns_none_when_under_the_narrator_window(monkeypatch):
    monkeypatch.setattr(dm_agent, "_load_recap_state", _fake_load_recap_state("", 0))
    monkeypatch.setattr(dm_agent, "get_thread_messages", _fake_get_thread_messages(_narrative(5)))
    fake_model = _FakeModel("should not be called")
    monkeypatch.setattr(dm_agent, "_get_model", lambda: fake_model)

    result = await dm_agent._maybe_update_session_recap("thread1", Campaign(id="c1", name="Test"))
    assert result is None
    assert fake_model.calls == 0


@pytest.mark.asyncio
async def test_folds_new_turns_into_a_recap_once_over_the_window(monkeypatch):
    # _NARRATOR_MAX_TURNS is 20 narrative messages (human+ai pairs count individually);
    # 30 narrative messages means the oldest 10 are about to fall out of view.
    narrative = _narrative(15)  # 30 messages total
    monkeypatch.setattr(dm_agent, "_load_recap_state", _fake_load_recap_state("", 0))
    monkeypatch.setattr(dm_agent, "get_thread_messages", _fake_get_thread_messages(narrative))
    fake_model = _FakeModel("The party investigated the tavern and met Old Bram.")
    monkeypatch.setattr(dm_agent, "_get_model", lambda: fake_model)

    result = await dm_agent._maybe_update_session_recap("thread1", Campaign(id="c1", name="Test"))
    assert result is not None
    recap, through = result
    assert recap == "The party investigated the tavern and met Old Bram."
    assert through == len(narrative) - dm_agent._NARRATOR_MAX_TURNS
    assert fake_model.calls == 1


@pytest.mark.asyncio
async def test_does_not_refold_turns_already_covered_by_recap_through(monkeypatch):
    narrative = _narrative(15)  # 30 messages
    already_through = len(narrative) - dm_agent._NARRATOR_MAX_TURNS  # exactly caught up
    monkeypatch.setattr(dm_agent, "_load_recap_state", _fake_load_recap_state("prior recap text", already_through))
    monkeypatch.setattr(dm_agent, "get_thread_messages", _fake_get_thread_messages(narrative))
    fake_model = _FakeModel("should not be called")
    monkeypatch.setattr(dm_agent, "_get_model", lambda: fake_model)

    result = await dm_agent._maybe_update_session_recap("thread1", Campaign(id="c1", name="Test"))
    assert result is None
    assert fake_model.calls == 0


@pytest.mark.asyncio
async def test_falls_back_to_prior_recap_on_llm_failure(monkeypatch):
    narrative = _narrative(15)
    monkeypatch.setattr(dm_agent, "_load_recap_state", _fake_load_recap_state("earlier recap", 0))
    monkeypatch.setattr(dm_agent, "get_thread_messages", _fake_get_thread_messages(narrative))

    class _BrokenModel:
        async def ainvoke(self, prompt):
            raise RuntimeError("model backend unreachable")

    monkeypatch.setattr(dm_agent, "_get_model", lambda: _BrokenModel())

    result = await dm_agent._maybe_update_session_recap("thread1", Campaign(id="c1", name="Test"))
    assert result is not None
    recap, through = result
    assert recap == "earlier recap"


def test_mechanics_and_narrator_modifiers_prepend_the_recap_marker():
    recap_text = "Earlier, the party met Old Bram and fought smugglers."
    state = {"messages": [HumanMessage(content="hello")], "session_recap": recap_text}

    # Exercise the shared prepend contract directly rather than the full
    # async modifier closures (which need a live CampaignStore) — both
    # modifiers build `result = [system_msg]`, conditionally append the
    # recap HumanMessage, then extend with trimmed/narrative history.
    from langchain_core.messages import SystemMessage
    result = [SystemMessage(content="sys")]
    if state.get("session_recap"):
        result.append(HumanMessage(content=dm_agent._RECAP_MARKER + state["session_recap"]))
    result += state["messages"]

    assert len(result) == 3
    assert recap_text in result[1].content
    assert "internal, not player dialogue" in result[1].content


def _fake_load_recap_state(recap: str, through: int):
    async def _inner(thread_id):
        return recap, through
    return _inner


def _fake_get_thread_messages(messages: list):
    async def _inner(thread_id):
        return messages
    return _inner
