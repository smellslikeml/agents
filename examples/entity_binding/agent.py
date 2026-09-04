from __future__ import annotations

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    function_tool,
    inference,
)
from livekit.agents.llm import BoundEntity, EntityCandidate, resolve_entity

logger = logging.getLogger("entity-binding")
load_dotenv()

# A tiny address book backing the ``send_email`` tool. Two people share a
# display name — the exact entity-binding hazard this example targets: a vague
# "email John Smith" must not be acted on, because it is unclear *which* one.
ADDRESS_BOOK: list[EntityCandidate] = [
    EntityCandidate(label="John Smith (Corp)", value="john@corp.example.com"),
    EntityCandidate(label="John Smith (Other)", value="john@other.example.com"),
    EntityCandidate(label="Maria Gomez", value="maria@example.com"),
]


def resolve_recipient(arguments: dict, call_ctx: object) -> dict[str, BoundEntity]:
    """Entity-resolution precondition for :func:`send_email`.

    Binds the ``to`` reference to exactly one address-book entry before the
    tool is allowed to run. :func:`~livekit.agents.llm.resolve_entity` raises
    :class:`~livekit.agents.llm.EntityBindingError` (a
    :class:`~livekit.agents.ToolError`) when the name is ambiguous — e.g. the
    two "John Smith" entries — or unmatched, which the framework routes back
    to the LLM as an error result so it asks the user to clarify instead of
    emailing the wrong person. On a unique match the resolved entry is
    returned as provenance.

    ref: https://arxiv.org/abs/2606.30531v1
    """
    del call_ctx  # binding is decided from the parsed arguments alone
    name = arguments.get("to") or ""
    bound = resolve_entity(name, ADDRESS_BOOK, arg_name="to", tool_name="send_email")
    return {"to": bound}


class EmailAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an assistant that can send emails on the user's behalf. "
                "When the user asks to send an email, call the send_email tool with "
                "the recipient's name and the subject. If send_email returns an "
                "error saying the recipient is ambiguous or unknown, do NOT guess — "
                "read back the candidate names it lists, ask the user which one they "
                "mean, and then call send_email again with the full, unambiguous name."
            )
        )

    @function_tool(entities=resolve_recipient)
    async def send_email(self, to: str, subject: str) -> str:
        """Send an email to a recipient.

        Args:
            to: the recipient's display name (e.g. "Maria Gomez").
            subject: the email subject line.
        """
        # We only reach here once the entity-binding precondition has
        # confirmed ``to`` matches exactly one address-book entry, so it is
        # safe to act on the reference. The resolved value/provenance is
        # attached to the resulting FunctionCallResult.entity_bindings.
        logger.info("sent email to %s: %s", to, subject)
        return f"Done — sent '{subject}' to {to}."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
    )
    await session.start(agent=EmailAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
