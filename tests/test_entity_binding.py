from __future__ import annotations

import json

import pytest

from livekit.agents.llm import (
    EntityBindingError,
    EntityCandidate,
    FunctionCallResult,
    FunctionToolCall,
    ToolContext,
    execute_function_call,
    function_tool,
    resolve_entity,
)

pytestmark = pytest.mark.unit


# A tiny "address book" backing the send_email tool — two people share a name,
# which is exactly the entity-binding hazard the paper targets.
ADDRESS_BOOK = [
    EntityCandidate(label="John Smith (Corp)", value="john@corp.example.com"),
    EntityCandidate(label="John Smith (Other)", value="john@other.example.com"),
    EntityCandidate(label="Maria Gomez", value="maria@example.com"),
]


def _email_resolver(arguments: dict, call_ctx: object) -> dict:
    """Entity-resolution precondition for send_email."""
    del call_ctx
    name = arguments.get("to") or ""
    bound = resolve_entity(name, ADDRESS_BOOK, arg_name="to", tool_name="send_email")
    return {"to": bound}


@function_tool(entities=_email_resolver)
def send_email(to: str, subject: str) -> str:
    """Send an email to a recipient.

    Args:
        to: the recipient's display name
        subject: the email subject
    """

    return f"sent '{subject}' to {to}"


@function_tool
def echo(text: str) -> str:
    """Echo back the text — no entity resolver declared."""

    return text


async def _run(tool_name: str, arguments: dict) -> FunctionCallResult:
    tc = FunctionToolCall(
        name=tool_name,
        arguments=json.dumps(arguments),
        call_id="c1",
    )
    tool_ctx = ToolContext([send_email, echo])
    return await execute_function_call(tc, tool_ctx)


async def test_unique_reference_binds_and_runs() -> None:
    result = await _run("send_email", {"to": "Maria Gomez", "subject": "hi"})

    assert result.fnc_call_out is not None
    assert result.fnc_call_out.is_error is False
    assert result.fnc_call_out.output == "sent 'hi' to Maria Gomez"
    # provenance is attached to the standalone-execution result
    assert [b.value for b in result.entity_bindings] == ["maria@example.com"]
    assert result.entity_bindings[0].arg_name == "to"


async def test_ambiguous_reference_is_fed_back_to_llm() -> None:
    # "John Smith" matches two address-book entries -> the tool must NOT run;
    # instead a clarification is returned as a tool error so the LLM can ask.
    result = await _run("send_email", {"to": "John Smith", "subject": "hi"})

    assert result.fnc_call_out is not None
    assert result.fnc_call_out.is_error is True
    output = result.fnc_call_out.output
    assert "John Smith (Corp)" in output
    assert "John Smith (Other)" in output
    assert "john@corp.example.com" not in output  # values are not leaked, only labels
    assert isinstance(result.raw_exception, EntityBindingError)
    assert result.raw_exception.kind == "ambiguous"


async def test_unknown_reference_is_fed_back_to_llm() -> None:
    result = await _run("send_email", {"to": "Nobody McUnknown", "subject": "hi"})

    assert result.fnc_call_out is not None
    assert result.fnc_call_out.is_error is True
    assert isinstance(result.raw_exception, EntityBindingError)
    assert result.raw_exception.kind == "not_found"


async def test_tool_without_resolver_is_unchanged() -> None:
    # No `entities=` declared -> the precondition is a no-op and behavior is
    # identical to before this feature existed.
    result = await _run("echo", {"text": "hello"})

    assert result.fnc_call_out is not None
    assert result.fnc_call_out.is_error is False
    assert result.fnc_call_out.output == "hello"
    assert result.entity_bindings == []


def test_resolve_entity_directly() -> None:
    bound = resolve_entity("maria", ADDRESS_BOOK, tool_name="t", arg_name="to")
    assert bound.value == "maria@example.com"

    with pytest.raises(EntityBindingError) as exc_info:
        resolve_entity("John Smith", ADDRESS_BOOK, tool_name="t", arg_name="to")
    assert exc_info.value.kind == "ambiguous"

    with pytest.raises(EntityBindingError) as exc_info:
        resolve_entity("Nobody McUnknown", ADDRESS_BOOK, tool_name="t", arg_name="to")
    assert exc_info.value.kind == "not_found"
