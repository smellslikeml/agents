from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable
from typing import TYPE_CHECKING

from .tool_search import BM25SearchStrategy, SearchItem, SearchStrategy

if TYPE_CHECKING:
    from ...llm.tool_context import Toolset

# Generic schema words that name no concrete entity, so two tools sharing them
# are not actually collaborating — they would only create spurious edges.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "for",
        "to",
        "and",
        "or",
        "in",
        "on",
        "with",
        "by",
        "from",
        "id",
        "ids",
        "name",
        "names",
        "value",
        "values",
        "data",
        "query",
        "string",
        "number",
        "type",
        "result",
        "input",
        "output",
        "arg",
        "args",
        "param",
        "params",
        "key",
        "field",
        "object",
        "item",
    }
)


def _collaboration_tokens(item: SearchItem) -> set[str]:
    """Tokens identifying the concrete entities a tool reads or writes.

    Collaboration is inferred from the tool *schema* — its parameter names and
    descriptions — because tools that operate on the same entity (e.g. both
    take a ``booking_id``) are the ones that compose into a task. Free-text name
    and description tokens are deliberately excluded: they are dominated by
    framework naming conventions (``*_tool``) and generic verbs that link
    unrelated tools.
    """
    tokens: list[str] = []
    for key, desc in item.parameters.items():
        tokens += key.lower().replace("_", " ").split()
        tokens += desc.lower().replace("_", " ").split()
    return {t for t in tokens if len(t) > 1 and t not in _STOPWORDS}


class ToolGraphSearchStrategy:
    """Intention-aware tool retrieval over a tool-collaboration graph.

    Adapted from *SING: Synthetic Intention Graph for Scalable Active Tool
    Discovery in LLM Agents* (arXiv:2606.16591). One-shot retrieval ranks each
    tool against the query in isolation, so it misses capabilities that only
    become relevant once a related tool is selected — exactly the long-horizon
    case where a task decomposes into collaborating sub-steps. SING addresses
    this by linking tools through their collaboration patterns and retrieving
    along those links as the task state evolves.

    This strategy brings that result to the existing :class:`SearchStrategy`
    contract without any external corpus or model: it wraps a base ranker
    (BM25 by default) and, at index time, builds a collaboration graph whose
    edges connect tools sharing distinctive (rare) schema entities or the
    same parent toolset. At search time the base ranker's relevance is diffused
    one hop across that graph, so a strongly-matched tool lifts its
    collaborators into the result set even when their own descriptions never
    mention the query. Results stay capped at ``max_results``, preserving the
    context-efficiency that motivates dynamic tool discovery.

    Args:
        base_strategy: Ranker used for the initial query/tool relevance. When
            omitted, a :class:`BM25SearchStrategy` is used.
        expansion_weight: Fraction of a tool's relevance propagated to each
            collaborator (0 disables expansion, recovering the base ranker).
        max_neighbors: Cap on edges kept per tool, bounding propagation cost on
            densely connected corpora.
        min_edge_weight: Minimum shared-token IDF mass required to form an edge.
    """

    def __init__(
        self,
        base_strategy: SearchStrategy | None = None,
        *,
        expansion_weight: float = 0.5,
        max_neighbors: int = 8,
        min_edge_weight: float = 0.0,
    ) -> None:
        self._base: SearchStrategy = base_strategy or BM25SearchStrategy()
        self._expansion_weight = expansion_weight
        self._max_neighbors = max_neighbors
        self._min_edge_weight = min_edge_weight
        # adjacency keyed by item identity -> list of (neighbor item, weight)
        self._edges: dict[int, list[tuple[SearchItem, float]]] = {}

    async def build_index(self, items: list[SearchItem]) -> None:
        result = self._base.build_index(items)
        if inspect.isawaitable(result):
            await result
        self._build_graph(items)

    async def search(
        self, query: str, items: list[SearchItem], max_results: int
    ) -> list[SearchItem]:
        # Pull the full base ranking so collaborators with zero direct
        # relevance can still be reached through graph diffusion.
        ranked = self._base.search(query, items, len(items))
        if inspect.isawaitable(ranked):
            ranked = await ranked
        ranked_list: list[SearchItem] = list(ranked)
        if not ranked_list:
            return []

        # Reciprocal-rank gives a bounded, base-agnostic relevance signal.
        base_score: dict[int, float] = {
            id(item): 1.0 / (rank + 1.0) for rank, item in enumerate(ranked_list)
        }

        final_score: dict[int, float] = dict(base_score)
        for item in items:
            neighbors = self._edges.get(id(item))
            if not neighbors:
                continue
            total_w = sum(w for _, w in neighbors)
            if total_w <= 0:
                continue
            boost = 0.0
            for neighbor, weight in neighbors:
                boost += base_score.get(id(neighbor), 0.0) * weight / total_w
            final_score[id(item)] = final_score.get(id(item), 0.0) + (
                self._expansion_weight * boost
            )

        scored = [(final_score.get(id(item), 0.0), idx, item) for idx, item in enumerate(items)]
        scored = [s for s in scored if s[0] > 0.0]
        # sort by score desc, original order as a stable tie-break
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [item for _, _, item in scored[:max_results]]

    def cleanup(self) -> None | Awaitable[None]:
        self._edges.clear()
        return self._base.cleanup()

    def _build_graph(self, items: list[SearchItem]) -> None:
        self._edges = {}
        if len(items) < 2:
            return

        token_sets = [_collaboration_tokens(item) for item in items]

        # document frequency -> IDF so common tokens contribute little weight
        df: dict[str, int] = {}
        for tokens in token_sets:
            for token in tokens:
                df[token] = df.get(token, 0) + 1
        n = len(items)
        idf = {token: math.log(n / freq) for token, freq in df.items()}

        sources: list[Toolset | None] = [_toolset_source(item) for item in items]

        for i in range(len(items)):
            neighbors: list[tuple[SearchItem, float]] = []
            for j in range(len(items)):
                if i == j:
                    continue
                shared = token_sets[i] & token_sets[j]
                weight = sum(idf.get(token, 0.0) for token in shared)
                # tools under the same toolset already load atomically, but the
                # edge lets matches in one toolset surface a sibling toolset's
                # tools when they also share capability tokens.
                if sources[i] is not None and sources[i] is sources[j]:
                    weight += 1.0
                if weight > self._min_edge_weight:
                    neighbors.append((items[j], weight))
            neighbors.sort(key=lambda nw: nw[1], reverse=True)
            if neighbors:
                self._edges[id(items[i])] = neighbors[: self._max_neighbors]


def _toolset_source(item: SearchItem) -> Toolset | None:
    from ...llm.tool_context import Toolset

    return item.source if isinstance(item.source, Toolset) else None
