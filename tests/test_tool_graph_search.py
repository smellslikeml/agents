import pytest

from livekit.agents.beta.toolsets.tool_graph_search import (
    ToolGraphSearchStrategy,
    _collaboration_tokens,
)
from livekit.agents.beta.toolsets.tool_search import (
    BM25SearchStrategy,
    SearchItem,
    ToolSearchToolset,
)
from livekit.agents.llm import Tool, ToolContext, Toolset, function_tool

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]


@function_tool
async def create_reservation(booking_id: str, guest_name: str) -> str:
    """Create a hotel reservation"""
    return "reserved"


@function_tool
async def charge_card(booking_id: str, amount: str) -> str:
    """Charge a payment card"""
    return "charged"


@function_tool
async def get_weather(city: str) -> str:
    """Get current weather"""
    return "sunny"


class BookingToolset(Toolset):
    def __init__(self) -> None:
        super().__init__(id="booking")

    @property
    def tools(self) -> list[Tool | Toolset]:
        return [create_reservation]


class BillingToolset(Toolset):
    def __init__(self) -> None:
        super().__init__(id="billing")

    @property
    def tools(self) -> list[Tool | Toolset]:
        return [charge_card]


class WeatherToolset(Toolset):
    def __init__(self) -> None:
        super().__init__(id="weather")

    @property
    def tools(self) -> list[Tool | Toolset]:
        return [get_weather]


def _make_items() -> list[SearchItem]:
    return [
        SearchItem(
            name="create_reservation",
            description="Create a hotel reservation",
            parameters={"booking_id": "Booking record identifier", "guest_name": "Guest name"},
            source=create_reservation,
        ),
        SearchItem(
            name="charge_card",
            description="Charge a payment card",
            parameters={"booking_id": "Booking record identifier", "amount": "Amount to charge"},
            source=charge_card,
        ),
        SearchItem(
            name="get_weather",
            description="Get current weather",
            parameters={"city": "City name"},
            source=get_weather,
        ),
    ]


class TestCollaborationTokens:
    def test_shared_schema_entity_extracted(self):
        items = _make_items()
        a = _collaboration_tokens(items[0])
        b = _collaboration_tokens(items[1])
        # both tools operate on a booking; that's the collaboration signal
        assert "booking" in a & b
        # generic schema words must not create edges
        assert "id" not in a

    def test_unrelated_tool_shares_nothing(self):
        items = _make_items()
        booking = _collaboration_tokens(items[0])
        weather = _collaboration_tokens(items[2])
        assert not (booking & weather)


class TestGraphExpansion:
    async def test_collaborator_surfaced_via_graph(self):
        """A query matching one tool should pull in a collaborator that shares a
        schema entity, even though the collaborator's own text never matches."""
        strategy = ToolGraphSearchStrategy()
        items = _make_items()
        await strategy.build_index(items)

        results = await strategy.search("reservation", items, 5)
        names = [r.name for r in results]
        assert "create_reservation" in names  # direct match
        assert "charge_card" in names  # surfaced through booking_id collaboration
        assert "get_weather" not in names  # no edge, no match

    async def test_no_expansion_recovers_base_ranking(self):
        items = _make_items()
        strategy = ToolGraphSearchStrategy(expansion_weight=0.0)
        await strategy.build_index(items)

        results = await strategy.search("reservation", items, 5)
        names = [r.name for r in results]
        assert names == ["create_reservation"]  # only the direct match remains

    async def test_isolated_match_does_not_pull_unrelated_tools(self):
        items = _make_items()
        strategy = ToolGraphSearchStrategy()
        await strategy.build_index(items)

        results = await strategy.search("weather", items, 5)
        names = [r.name for r in results]
        assert names == ["get_weather"]

    async def test_empty_query_returns_nothing(self):
        items = _make_items()
        strategy = ToolGraphSearchStrategy()
        await strategy.build_index(items)
        assert await strategy.search("", items, 5) == []

    async def test_cleanup_clears_graph_and_base(self):
        items = _make_items()
        base = BM25SearchStrategy()
        strategy = ToolGraphSearchStrategy(base)
        await strategy.build_index(items)
        assert strategy._edges

        strategy.cleanup()
        assert not strategy._edges
        assert not base._idf


class TestToolSearchToolsetUsesGraphStrategy:
    """Exercises the wiring edit in tool_search.ToolSearchToolset."""

    def test_default_strategy_is_graph(self):
        ts = ToolSearchToolset(id="search", tools=[BookingToolset()])
        assert isinstance(ts._strategy, ToolGraphSearchStrategy)

    async def test_search_loads_collaborating_toolset(self):
        ts = ToolSearchToolset(
            id="search",
            tools=[BookingToolset(), BillingToolset(), WeatherToolset()],
        )
        await ts.setup()

        # "reservation" only names the booking tool; the billing toolset is
        # reached because charge_card shares the booking_id entity.
        await ts._handle_search({"query": "reservation"})
        ctx = ToolContext([ts])
        assert "create_reservation" in ctx.function_tools
        assert "charge_card" in ctx.function_tools
        assert "get_weather" not in ctx.function_tools
