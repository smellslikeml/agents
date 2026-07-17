from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .agent_session import AgentSession


__all__ = [
    "TOOL_LEDGER_KEY",
    "LedgerEntry",
    "ToolCallLedger",
    "attach_ledger",
    "get_ledger",
]


# Key under which a ``ToolCallLedger`` is hosted when ``session.userdata`` is a
# plain dict. A ledger may also be attached *as* the userdata itself.
TOOL_LEDGER_KEY = "tool_ledger"


@dataclass
class LedgerEntry:
    """One recorded tool call — a piece of structured task state.

    Attributes:
        call_id: The function-call identifier shared with the model.
        name: The tool name (``FunctionTool.info.name``).
        arguments: The raw arguments the call was dispatched with.
        status: ``"running"`` until the call resolves.
        result: The tool's return value once ``status == "completed"``.
        error: Human-readable error once ``status == "failed"``.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]
    status: Literal["running", "completed", "failed"] = "running"
    result: Any | None = None
    error: str | None = None


class ToolCallLedger:
    """Session-scoped, append-only structured-state ledger of tool calls.

    Adapted from LedgerAgent (arxiv:2606.20529), which keeps observed task
    state in a separate ledger rendered into the agent prompt instead of
    leaving the model to reconstruct it from raw prompt context each turn.

    This ships only the *structured-state* data structure: every tool call
    flowing through the voice ``_ToolExecutor`` is appended as a typed
    ``LedgerEntry``. The paper's policy-adherence layer — consuming the
    ledger to block state-dependent policy violations before
    environment-changing calls execute — is intentionally out of scope here;
    it earns its own integration when a caller needs it.

    Attach to a session with :func:`attach_ledger` (or by placing an instance
    on ``session.userdata``); the executor reads it through :func:`get_ledger`
    and silently no-ops when none is attached.
    """

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._by_id: dict[str, LedgerEntry] = {}

    def record_call(self, call_id: str, name: str, arguments: dict[str, Any]) -> LedgerEntry:
        """Append a ``running`` entry for a freshly dispatched call."""
        entry = LedgerEntry(call_id=call_id, name=name, arguments=dict(arguments))
        self._entries.append(entry)
        self._by_id[call_id] = entry
        return entry

    def record_result(self, call_id: str, output: Any) -> None:
        """Resolve a pending entry. A ``BaseException`` output marks it failed."""
        entry = self._by_id.get(call_id)
        if entry is None:
            return
        if isinstance(output, BaseException):
            entry.status = "failed"
            entry.error = str(output) or type(output).__name__
        else:
            entry.status = "completed"
            entry.result = output

    def by_name(self, name: str) -> list[LedgerEntry]:
        """All entries for a given tool name, in insertion order."""
        return [entry for entry in self._entries if entry.name == name]

    def failed(self) -> list[LedgerEntry]:
        """All entries whose call raised (status ``"failed"``)."""
        return [entry for entry in self._entries if entry.status == "failed"]

    def entries(self) -> list[LedgerEntry]:
        """A shallow copy of every recorded entry, in insertion order."""
        return list(self._entries)

    def summary(self) -> str:
        """A human-readable, prompt-renderable snapshot of recorded state."""
        if not self._entries:
            return "No tool calls recorded."
        lines = [f"{len(self._entries)} tool call(s):"]
        for entry in self._entries:
            if entry.status == "completed":
                lines.append(f"  - {entry.name} ({entry.call_id}): completed")
            elif entry.status == "failed":
                lines.append(f"  - {entry.name} ({entry.call_id}): failed — {entry.error}")
            else:
                lines.append(f"  - {entry.name} ({entry.call_id}): running")
        return "\n".join(lines)


def get_ledger(session: AgentSession[Any]) -> ToolCallLedger | None:
    """Return the ledger attached to ``session``, or ``None`` if none is.

    Reads ``session.userdata`` safely: an unset userdata (the common case for
    sessions that don't opt into the ledger) yields ``None`` rather than the
    underlying ``ValueError``. Recognizes a ledger stored directly as the
    userdata or hosted under :data:`TOOL_LEDGER_KEY` in a dict userdata.
    """
    try:
        userdata = session.userdata
    except ValueError:
        return None
    if isinstance(userdata, ToolCallLedger):
        return userdata
    if isinstance(userdata, dict):
        hosted = userdata.get(TOOL_LEDGER_KEY)
        if isinstance(hosted, ToolCallLedger):
            return hosted
    return None


def attach_ledger(session: AgentSession[Any], ledger: ToolCallLedger) -> None:
    """Attach ``ledger`` to ``session.userdata`` under :data:`TOOL_LEDGER_KEY`.

    Creates a dict userdata when none is set, or adds the key to an existing
    dict userdata. Raises ``TypeError`` if userdata is already a non-dict value
    the caller cares about — in that case host the ledger yourself rather than
    letting this helper clobber it.
    """
    try:
        userdata = session.userdata
    except ValueError:
        userdata = None

    if userdata is None:
        session.userdata = {TOOL_LEDGER_KEY: ledger}
    elif isinstance(userdata, dict):
        userdata[TOOL_LEDGER_KEY] = ledger
    else:
        raise TypeError(
            f"session.userdata is already {type(userdata).__name__!r}; refusing to "
            f"clobber it. Host the ToolCallLedger at session.userdata[{TOOL_LEDGER_KEY!r}] "
            f"yourself."
        )
