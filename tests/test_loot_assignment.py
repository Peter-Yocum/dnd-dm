"""BDD-style coverage for the deterministic loot-assignment feature (see
backend/tools/_helpers.py's assign_container_item/assign_container_currency,
and the GET/POST /campaigns/{id}/loot* routes in backend/main.py) — built to
guard against the "narrated loot never actually reached anyone's inventory"
bug this feature exists to route around entirely.
"""
import pytest

from backend.models import Container, Currency, Item
from backend.tools._helpers import (
    assign_container_currency, assign_container_item, find_container_by_id,
)

from tests.conftest import make_character

pytestmark = pytest.mark.asyncio


async def test_assigning_an_item_moves_it_into_the_characters_inventory(store, campaign):
    # Given a shared find sitting unclaimed in a container
    char = make_character("Tarvokk")
    container = Container(
        name="the fallen enemies",
        contents=[Item(name="Shortbow"), Item(name="Quiver of 10 Arrows")],
        currency=Currency(gp=2),
    )
    campaign.party = [char]
    campaign.containers = [container]
    await store.save(campaign)

    # When one item is assigned to a character
    reloaded = await store.load(campaign.id)
    reloaded_char = reloaded.party[0]
    reloaded_container = find_container_by_id(reloaded, container.id)
    item_id = next(i.id for i in reloaded_container.contents if i.name == "Shortbow")
    summary = assign_container_item(reloaded, reloaded_container, item_id, reloaded_char)
    await store.save(reloaded)

    # Then it's really in the character's inventory, and gone from the
    # container — verified via a fresh reload, not the in-memory mutation
    assert "moved to Tarvokk's inventory" in summary
    final = await store.load(campaign.id)
    final_char = final.party[0]
    assert any(i.name == "Shortbow" for i in final_char.inventory)
    final_container = find_container_by_id(final, container.id)
    assert not any(i.name == "Shortbow" for i in final_container.contents)
    # The quiver and the gold are still unclaimed — container isn't dropped
    # until everything in it has been assigned
    assert any(i.name == "Quiver of 10 Arrows" for i in final_container.contents)


async def test_assigning_the_last_item_and_currency_removes_the_empty_container(store, campaign):
    # Given a container with exactly one item and some gold
    char = make_character("Tarvokk")
    container = Container(name="a small pouch", contents=[Item(name="Whetstone")], currency=Currency(gp=5))
    campaign.party = [char]
    campaign.containers = [container]
    await store.save(campaign)

    reloaded = await store.load(campaign.id)
    reloaded_char = reloaded.party[0]
    reloaded_container = find_container_by_id(reloaded, container.id)
    item_id = reloaded_container.contents[0].id

    # When both the item and all the currency are assigned away
    assign_container_item(reloaded, reloaded_container, item_id, reloaded_char)
    assign_container_currency(reloaded, reloaded_container, "gp", 5, reloaded_char)
    await store.save(reloaded)

    # Then the character actually has both, and the now-empty container is
    # gone from the campaign entirely (nothing left to claim)
    final = await store.load(campaign.id)
    final_char = final.party[0]
    assert any(i.name == "Whetstone" for i in final_char.inventory)
    assert final_char.currency.gp == 5
    assert find_container_by_id(final, container.id) is None


async def test_assigning_currency_caps_at_whats_actually_in_the_container(store, campaign):
    # Given a container with only 3 gp in it
    char = make_character("Tarvokk")
    container = Container(name="a coin purse", currency=Currency(gp=3))
    campaign.party = [char]
    campaign.containers = [container]
    await store.save(campaign)

    reloaded = await store.load(campaign.id)
    reloaded_char = reloaded.party[0]
    reloaded_container = find_container_by_id(reloaded, container.id)

    # When a request tries to claim more gold than actually exists
    summary = assign_container_currency(reloaded, reloaded_container, "gp", 999, reloaded_char)
    await store.save(reloaded)

    # Then only what was really there gets moved, not the requested amount
    assert "3 gp moved" in summary
    final_char = (await store.load(campaign.id)).party[0]
    assert final_char.currency.gp == 3
