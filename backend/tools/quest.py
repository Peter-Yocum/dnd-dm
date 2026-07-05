from langchain_core.tools import tool

from backend.models import QuestStatus
from backend.stores.campaign_store import CampaignStore
from backend.tools._helpers import find_quest


def make_tools(campaign_id: str, store: CampaignStore) -> list:

    @tool
    async def get_active_quests() -> str:
        """List all active quests with their objectives and progress. Call when
        the party checks their quest log or before awarding completion rewards."""
        campaign = await store.load(campaign_id)
        active = [q for q in campaign.quests if q.status == QuestStatus.ACTIVE]
        if not active:
            return "No active quests."
        lines = []
        for q in active:
            lines.append(f"── {q.name} [{q.quest_type.value}] ──")
            if q.description:
                lines.append(f"  {q.description}")
            if q.giver:
                lines.append(f"  Quest giver: {q.giver}")
            if q.objectives:
                lines.append("  Objectives:")
                for obj in q.objectives:
                    mark = "✓" if obj.is_completed else "○"
                    lines.append(f"    {mark} {obj.description}")
            r = q.rewards
            reward_parts = []
            if r.xp: reward_parts.append(f"{r.xp} XP")
            if r.gold: reward_parts.append(f"{r.gold} gp")
            if r.items: reward_parts.append(", ".join(i.name for i in r.items))
            if reward_parts:
                lines.append("  Rewards: " + ", ".join(reward_parts))
            if q.notes:
                lines.append(f"  Notes: {q.notes}")
        return "\n".join(lines)

    @tool
    async def complete_quest_objective(quest_name: str, objective_index: int) -> str:
        """Mark a single quest objective as completed. Use the 0-based index
        from get_active_quests. Does not complete the whole quest — use
        change_quest_status for that."""
        campaign = await store.load(campaign_id)
        quest = find_quest(campaign, quest_name)
        if not quest:
            return f"No quest named '{quest_name}' found."
        if not (0 <= objective_index < len(quest.objectives)):
            return f"Quest '{quest_name}' has no objective at index {objective_index}."
        obj = quest.objectives[objective_index]
        if obj.is_completed:
            return f"Objective already completed: '{obj.description}'."
        obj.is_completed = True
        await store.save(campaign)
        remaining = sum(1 for o in quest.objectives if not o.is_completed)
        msg = f"Objective completed: '{obj.description}'."
        if remaining == 0:
            msg += f" All objectives done — consider completing quest '{quest_name}'."
        else:
            msg += f" {remaining} objective(s) remaining."
        return msg

    @tool
    async def change_quest_status(quest_name: str, status: str) -> str:
        """Change a quest's status. Valid values: unknown, active, completed, failed.
        Use 'completed' when the party finishes a quest, 'failed' if it expires or
        they choose to abandon it, 'active' to pick up a quest they've just accepted."""
        campaign = await store.load(campaign_id)
        quest = find_quest(campaign, quest_name)
        if not quest:
            return f"No quest named '{quest_name}' found."
        try:
            new_status = QuestStatus(status.lower())
        except ValueError:
            return f"'{status}' is not a valid status. Use: {', '.join(s.value for s in QuestStatus)}."
        old = quest.status.value
        quest.status = new_status
        await store.save(campaign)
        msg = f"Quest '{quest.name}': {old} → {new_status.value}."
        if new_status == QuestStatus.COMPLETED:
            r = quest.rewards
            parts = []
            if r.xp: parts.append(f"{r.xp} XP")
            if r.gold: parts.append(f"{r.gold} gp")
            if r.items: parts.append(", ".join(i.name for i in r.items))
            if parts:
                msg += f" Rewards due: {', '.join(parts)}."
        return msg

    return [get_active_quests, complete_quest_objective, change_quest_status]
