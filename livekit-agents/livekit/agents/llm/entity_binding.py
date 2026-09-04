# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entity binding guards for tool-augmented agents.

Adapted from *Entity Binding Failures in Tool-Augmented Agents*
(https://arxiv.org/abs/2606.30531v1).

A tool-augmented agent may select the *correct* tool and produce *valid*
arguments and still act on the *wrong* real-world entity — e.g. emailing the
wrong "Alex" or booking the wrong "John Smith".  This module formalizes that
failure as a distinct reliability problem and ships the building blocks the
paper evaluates:

* :class:`EntityBindingError` — a structured :class:`~livekit.agents.llm.ToolError`
  raised when a natural-language reference cannot be confidently bound to
  exactly one entity.  Because it subclasses ``ToolError``, the framework's
  existing tool-error feedback loop
  (:func:`~livekit.agents.llm.utils.make_function_call_output` →
  ``FunctionCallOutput(is_error=True)``) feeds the clarification straight back
  to the LLM so it can self-correct — the same channel the "unknown function,
  available tools" message uses.
* :func:`resolve_entity` — confidence-gated binding with clarification under
  ambiguity.  Returns a :class:`BoundEntity` (provenance) or raises.
* :func:`run_tool_entity_resolver` — the precondition hook invoked by
  :func:`~livekit.agents.llm.utils.execute_function_call` between argument
  preparation and the actual call, so an opt-in
  ``@function_tool(entities=...)`` is guarded before it ever runs.

This module deliberately ships the *result* (reliable binding + clarification
feedback), not the paper's full failure taxonomy, provenance store, or any
training procedure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from .tool_context import ToolError

if TYPE_CHECKING:
    from ..voice.events import RunContext
    from .tool_context import FunctionTool, RawFunctionTool

__all__ = [
    "DEFAULT_ENTITY_THRESHOLD",
    "BindingKind",
    "EntityCandidate",
    "BoundEntity",
    "EntityResolver",
    "EntityBindingError",
    "resolve_entity",
    "run_tool_entity_resolver",
]

DEFAULT_ENTITY_THRESHOLD: float = 0.6
"""Minimum normalized similarity (0.0–1.0) for a candidate to count as a match.

Lower it to bind fuzzier references; raise it to demand closer names before
acting. Tune per tool — entity-critical tools (payments, account mutations)
should run a stricter threshold than lookups."""

BindingKind = Literal["ambiguous", "not_found"]


@dataclass
class EntityCandidate:
    """A real-world entity the agent could act on."""

    label: str
    """Natural-language label used to match the reference (e.g. a display name)."""
    value: Any
    """The underlying entity to bind (surfaced in :class:`BoundEntity`)."""


@dataclass
class BoundEntity:
    """Provenance for a resolved entity binding."""

    arg_name: str
    reference: str
    value: Any
    score: float
    candidates: list[EntityCandidate] = field(default_factory=list)
    """Every candidate that was considered for this reference."""


EntityResolver: TypeAlias = Callable[
    [dict[str, Any], "RunContext[Any] | None"], dict[str, BoundEntity]
]
"""Resolves entity-typed arguments for a tool.

Given the parsed ``arguments`` and the optional ``call_ctx`` (e.g. a knowledge
base reachable from the run context), return a mapping of argument name →
:class:`BoundEntity`.  Raise :class:`EntityBindingError` when a reference is
ambiguous or unmatched, so the framework asks the LLM to clarify instead of
acting on the wrong entity."""


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _as_candidate(candidate: EntityCandidate | str) -> EntityCandidate:
    if isinstance(candidate, EntityCandidate):
        return candidate
    return EntityCandidate(label=candidate, value=candidate)


def _score(reference_norm: str, label_norm: str) -> float:
    if not reference_norm or not label_norm:
        return 0.0
    return SequenceMatcher(None, reference_norm, label_norm).ratio()


class EntityBindingError(ToolError):
    """Raised when a reference cannot be confidently bound to one entity.

    Subclasses :class:`~livekit.agents.llm.ToolError`, so the framework routes
    ``message`` back to the LLM as an error result (``is_error=True``), listing
    the candidate entities so the model can disambiguate on its next attempt —
    mirroring the "available tools" self-correct message for unknown functions.
    """

    def __init__(
        self,
        tool_name: str,
        arg_name: str,
        reference: str,
        candidates: Sequence[EntityCandidate | str],
        *,
        kind: BindingKind,
    ) -> None:
        self.tool_name = tool_name
        self.arg_name = arg_name
        self.reference = reference
        self.kind = kind
        self.candidates: list[EntityCandidate] = [_as_candidate(c) for c in candidates]
        super().__init__(self._render_message())

    def _render_message(self) -> str:
        location = f"tool `{self.tool_name}`"
        if self.arg_name:
            location += f" argument `{self.arg_name}`"
        names = "\n".join(f" - {c.label}" for c in self.candidates)
        if self.kind == "ambiguous":
            intro = (
                f"Entity binding for {location} is ambiguous: the reference "
                f'"{self.reference}" matches more than one entity. '
                "Ask the user which one they mean before calling again."
            )
            tail = "Matching entities:\n" + names if names else ""
        else:
            intro = (
                f"Entity binding for {location} failed: no entity matches the "
                f'reference "{self.reference}" closely enough. '
                "Confirm the name with the user before calling again."
            )
            tail = "Known entities:\n" + names if names else ""
        return "\n".join(part for part in (intro, tail) if part)


def resolve_entity(
    reference: str,
    candidates: Sequence[EntityCandidate | str],
    *,
    arg_name: str = "",
    tool_name: str = "",
    threshold: float = DEFAULT_ENTITY_THRESHOLD,
) -> BoundEntity:
    """Bind ``reference`` to exactly one candidate, or raise.

    Scores each candidate label against the reference (normalized, case-folded
    fuzzy ratio) and applies confidence-gated binding:

    * exactly one candidate clears ``threshold`` → bound;
    * two or more clear it → :class:`EntityBindingError` (``kind="ambiguous"``);
    * none clear it → :class:`EntityBindingError` (``kind="not_found"``).

    The returned :class:`BoundEntity` carries provenance: the resolved value,
    its score, and every candidate considered.
    """
    considered = [_as_candidate(c) for c in candidates]
    ref_norm = _normalize(reference)

    scored: list[tuple[float, EntityCandidate]] = []
    matches: list[EntityCandidate] = []
    if considered and ref_norm:
        scored = sorted(
            ((_score(ref_norm, _normalize(c.label)), c) for c in considered),
            key=lambda pair: pair[0],
            reverse=True,
        )
        matches = [c for score, c in scored if score >= threshold]

    if len(matches) == 1:
        best_score, best = scored[0]
        return BoundEntity(
            arg_name=arg_name,
            reference=reference,
            value=best.value,
            score=best_score,
            candidates=considered,
        )

    if len(matches) > 1:
        raise EntityBindingError(tool_name, arg_name, reference, matches, kind="ambiguous")

    raise EntityBindingError(tool_name, arg_name, reference, considered, kind="not_found")


def run_tool_entity_resolver(
    tool: FunctionTool | RawFunctionTool,
    arguments: dict[str, Any],
    call_ctx: RunContext[Any] | None = None,
) -> list[BoundEntity]:
    """Run a tool's declared entity resolver as an execution precondition.

    Returns provenance for every resolved argument.  When the resolver raises
    :class:`EntityBindingError`, it propagates so the caller
    (:func:`~livekit.agents.llm.utils.execute_function_call`) routes the
    clarification back to the LLM through the existing tool-error feedback loop.

    Tools that do not declare an entity resolver are a no-op (returns ``[]``),
    preserving existing behavior.
    """
    resolver = getattr(tool.info, "entities", None)
    if resolver is None:
        return []
    bindings = resolver(arguments, call_ctx)
    return list(bindings.values())
