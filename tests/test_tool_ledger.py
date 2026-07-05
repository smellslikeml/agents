from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents import function_tool
from livekit.agents.llm import FunctionCall
from livekit.agents.voice.events import RunContext
from livekit.agents.voice.tool_ledger import (
    TOOL_LEDGER_KEY,
    LedgerEntry,
    ToolCallLedger,
    attach_ledger,
    get_ledger,
)

pytestmark = pytest.mark.unit


# --- RunContext helper (mirrors the one in tests/test_tools.py) ---------------


def _make_run_context(
    call_id: str = "call_1",
    name: str = "test_tool",
    arguments: str = "{}",
    allow_interruptions: bool = True,
) -> RunContext:
    """Build a minimal RunContext — only what the executor actually reads."""
    speech_handle = MagicMock()
    speech_handle.num_steps = 1
    speech_handle.allow_interruptions = allow_interruptions

    fnc_call = FunctionCall(call_id=call_id, name=name, arguments=arguments, extra={})
    return RunContext(
        session=MagicMock(),
        speech_handle=speech_handle,
        function_call=fnc_call,
    )


@pytest.fixture
def _clear_running_tasks():
    """Wipe the module-level registry between tests to avoid cross-test bleed."""
    from livekit.agents.voice.tool_executor import _RunningTasks

    _RunningTasks.clear()
    yield
    _RunningTasks.clear()


# --- ledger data structure ----------------------------------------------------


class TestToolCallLedger:
    def test_record_call_and_result_yields_completed_entry(self) -> None:
        ledger = ToolCallLedger()
        entry = ledger.record_call("c1", "lookup", {"q": "x"})

        assert isinstance(entry, LedgerEntry)
        assert entry.status == "running"
        assert entry.arguments == {"q": "x"}

        ledger.record_result("c1", {"found": True})

        assert entry.status == "completed"
        assert entry.result == {"found": True}
        assert entry.error is None

    def test_record_result_with_exception_marks_failed(self) -> None:
        ledger = ToolCallLedger()
        ledger.record_call("c2", "charge", {"amount": 5})
        ledger.record_result("c2", ValueError("insufficient funds"))

        entry = ledger.by_name("charge")[0]
        assert entry.status == "failed"
        assert entry.error == "insufficient funds"
        assert entry.result is None

    def test_record_result_unknown_call_id_is_noop(self) -> None:
        ledger = ToolCallLedger()
        ledger.record_result("never_recorded", "ok")  # must not raise
        assert ledger.entries() == []

    def test_by_name_and_failed_filter_across_calls(self) -> None:
        ledger = ToolCallLedger()
        ledger.record_call("a", "search", {"q": 1})
        ledger.record_call("b", "search", {"q": 2})
        ledger.record_call("c", "refund", {})
        ledger.record_result("a", "ok")
        ledger.record_result("b", RuntimeError("boom"))
        # "c" stays running

        assert [e.call_id for e in ledger.by_name("search")] == ["a", "b"]
        assert [e.call_id for e in ledger.failed()] == ["b"]
        assert ledger.by_name("refund")[0].status == "running"

    def test_summary_lists_each_call_with_status(self) -> None:
        ledger = ToolCallLedger()
        assert ledger.summary() == "No tool calls recorded."
        ledger.record_call("a", "search", {})
        ledger.record_result("a", "ok")
        ledger.record_call("b", "refund", {})
        summary = ledger.summary()
        assert "2 tool call(s)" in summary
        assert "search (a): completed" in summary
        assert "refund (b): running" in summary

    def test_record_call_snapshots_arguments(self) -> None:
        """The recorded arguments must not alias a dict the caller later mutates."""
        ledger = ToolCallLedger()
        args = {"q": "x"}
        ledger.record_call("c", "search", args)
        args["q"] = "mutated"

        assert ledger.entries()[0].arguments == {"q": "x"}


# --- attach / get helpers -----------------------------------------------------


class _FakeSession:
    """Minimal session exposing the real ``userdata`` property semantics."""

    def __init__(self) -> None:
        self._userdata: Any = None

    @property
    def userdata(self) -> Any:
        if self._userdata is None:
            raise ValueError("AgentSession userdata is not set")
        return self._userdata

    @userdata.setter
    def userdata(self, value: Any) -> None:
        self._userdata = value


class TestAttachAndGet:
    def test_get_returns_none_when_userdata_unset(self) -> None:
        assert get_ledger(_FakeSession()) is None  # type: ignore[arg-type]

    def test_attach_into_fresh_session_then_get(self) -> None:
        session = _FakeSession()
        ledger = ToolCallLedger()
        attach_ledger(session, ledger)  # type: ignore[arg-type]

        assert session.userdata[TOOL_LEDGER_KEY] is ledger
        assert get_ledger(session) is ledger  # type: ignore[arg-type]

    def test_attach_into_existing_dict_userdata_preserves_other_keys(self) -> None:
        session = _FakeSession()
        session.userdata = {"existing": 42}
        ledger = ToolCallLedger()
        attach_ledger(session, ledger)  # type: ignore[arg-type]

        assert session.userdata["existing"] == 42
        assert get_ledger(session) is ledger  # type: ignore[arg-type]

    def test_get_recognizes_ledger_hosted_as_userdata_directly(self) -> None:
        session = _FakeSession()
        ledger = ToolCallLedger()
        session.userdata = ledger
        assert get_ledger(session) is ledger  # type: ignore[arg-type]

    def test_get_returns_none_for_unrelated_userdata(self) -> None:
        session = _FakeSession()
        session.userdata = {"unrelated": 1}
        assert get_ledger(session) is None  # type: ignore[arg-type]

    def test_attach_refuses_to_clobber_non_dict_userdata(self) -> None:
        session = _FakeSession()
        session.userdata = object()  # some opaque user value
        with pytest.raises(TypeError, match="refusing to clobber"):
            attach_ledger(session, ToolCallLedger())  # type: ignore[arg-type]


# --- integration: drives the real _ToolExecutor.execute() call site ----------


class TestExecutorIntegration:
    """Exercises the wiring in ``voice/tool_executor.py``: an attached ledger
    records matching entries for every dispatched call; an unattached one
    leaves execution unchanged."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_execute_records_call_and_result_when_ledger_attached(self) -> None:
        from livekit.agents.voice.tool_executor import _ToolExecutor

        @function_tool
        async def balance_tool(account: str) -> str:
            """Get the balance for an account."""
            return f"balance:{account}"

        executor = _ToolExecutor()
        ledger = ToolCallLedger()
        run_ctx = _make_run_context(call_id="b1", name="balance_tool", arguments='{"account":"a1"}')
        run_ctx._session.userdata = ledger  # attach as the session's userdata

        result = await executor.execute(
            tool=balance_tool, run_ctx=run_ctx, raw_arguments={"account": "a1"}
        )
        assert result == "balance:a1"
        await asyncio.sleep(0)  # let _on_done fire

        entries = ledger.entries()
        assert len(entries) == 1
        assert entries[0].call_id == "b1"
        assert entries[0].name == "balance_tool"
        assert entries[0].arguments == {"account": "a1"}
        assert entries[0].status == "completed"
        assert entries[0].result == "balance:a1"

    @pytest.mark.asyncio
    async def test_execute_records_failure_when_tool_raises(self) -> None:
        from livekit.agents.llm.tool_context import ToolError
        from livekit.agents.voice.tool_executor import _ToolExecutor

        @function_tool
        async def boom_tool() -> str:
            """Always fails."""
            raise ToolError("kaboom")

        executor = _ToolExecutor()
        ledger = ToolCallLedger()
        run_ctx = _make_run_context(call_id="x1", name="boom_tool")
        run_ctx._session.userdata = ledger

        with pytest.raises(ToolError, match="kaboom"):
            await executor.execute(tool=boom_tool, run_ctx=run_ctx, raw_arguments={})
        await asyncio.sleep(0)

        entries = ledger.entries()
        assert len(entries) == 1
        assert entries[0].status == "failed"
        assert "kaboom" in (entries[0].error or "")
        assert [e.call_id for e in ledger.failed()] == ["x1"]

    @pytest.mark.asyncio
    async def test_execute_is_unchanged_when_no_ledger_attached(self) -> None:
        from livekit.agents.voice.tool_executor import _ToolExecutor

        @function_tool
        async def quick_tool() -> str:
            """q"""
            return "ok"

        executor = _ToolExecutor()
        # default MagicMock session — get_ledger returns None, execution no-ops
        run_ctx = _make_run_context(call_id="q1", name="quick_tool")

        result = await executor.execute(tool=quick_tool, run_ctx=run_ctx, raw_arguments={})
        assert result == "ok"
