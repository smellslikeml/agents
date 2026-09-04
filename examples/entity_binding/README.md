# Entity Binding Example

A voice agent that guards tool calls against **entity-binding failures** — the
case where an agent picks the correct tool and valid arguments but resolves the
*wrong real-world entity* (e.g. emailing the wrong "John Smith").

For setup instructions and more details, see the [main examples README](../README.md).

## Overview

`send_email` is declared with an entity-resolution precondition:

```python
@function_tool(entities=resolve_recipient)
async def send_email(self, to: str, subject: str) -> str: ...
```

Before the tool is allowed to run, `resolve_recipient` binds the `to` reference
to exactly one address-book entry via
[`resolve_entity`](https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/llm/entity_binding.py):

- **unique match** → the tool runs and the resolved entry is attached to the
  result as provenance (`FunctionCallResult.entity_bindings`);
- **ambiguous match** (two entries clear the threshold) → raises
  `EntityBindingError`, which the framework feeds back to the LLM as an error
  result listing the candidates — so the agent asks the user to clarify instead
  of guessing;
- **no match** → raises `EntityBindingError`, prompting the agent to confirm the
  name with the user.

The address book deliberately contains two people who share a display name, so
the disambiguation path is exercised:

```python
ADDRESS_BOOK = [
    EntityCandidate(label="John Smith (Corp)", value="john@corp.example.com"),
    EntityCandidate(label="John Smith (Other)", value="john@other.example.com"),
    EntityCandidate(label="Maria Gomez", value="maria@example.com"),
]
```

### Try it

- *"Send Maria Gomez an email saying hi."* → unique match, the email is sent.
- *"Email John Smith about the meeting."* → ambiguous; the agent replies with
  both `John Smith (Corp)` and `John Smith (Other)` and asks which one you mean
  before calling the tool again.

`EntityBindingError` subclasses `ToolError`, so the clarification reuses the
repo's existing tool-error feedback loop (`make_function_call_output` →
`FunctionCallOutput(is_error=True)`) — the same self-correct channel already used
for "unknown function, available tools."

This formalizes the reliability guarantee from *Entity Binding Failures in
Tool-Augmented Agents* ([arXiv:2606.30531](https://arxiv.org/abs/2606.30531v1)).
