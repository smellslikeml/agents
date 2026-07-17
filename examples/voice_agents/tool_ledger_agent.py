"""Minimal voice agent that opts a session into the tool-call ledger.

Attaching a ``ToolCallLedger`` (LedgerAgent, arxiv:2606.20529) makes every tool
call flowing through the executor append a typed entry — no per-tool bookkeeping
is required. This example attaches one and logs its ``summary()`` at shutdown.
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    inference,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import ToolCallLedger, attach_ledger

logger = logging.getLogger("tool-ledger-agent")

load_dotenv()


class LedgerAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an assistant that interacts with users via voice. "
                "Keep your responses concise and to the point. "
                "Do not use emojis, asterisks, or markdown."
            ),
        )

    @function_tool
    async def lookup_balance(self, ctx: RunContext, account: str) -> str:
        """Called when the user asks for an account balance.

        Args:
            account: The account identifier to look up.
        """
        logger.info("Looking up balance for %s", account)
        # This call is recorded automatically once a ledger is attached below —
        # no extra bookkeeping is needed inside the tool itself.
        return f"{account} has a balance of $1,000."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    # Opt the session into structured-state recording. The executor reads the
    # ledger through get_ledger(session) and silently no-ops for sessions that
    # haven't attached one, so existing behavior is unchanged without this.
    ledger = ToolCallLedger()

    session = AgentSession(
        # See https://docs.livekit.io/agents/models/stt/
        stt=inference.STT("deepgram/nova-3", language="multi"),
        # See https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM("openai/gpt-4.1-mini"),
        # See all voices at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    )

    # Host the ledger on session.userdata so the executor can find it. This
    # creates a dict userdata when none is set, or adds the key to an existing
    # dict without clobbering anything else you stored there.
    attach_ledger(session, ledger)

    def log_ledger() -> None:
        # summary() is the prompt-renderable snapshot of recorded tool state.
        # Automatic injection into the LLM prompt is intentionally deferred; here
        # we just log it at shutdown to show what was captured during the call.
        logger.info("Tool ledger at shutdown:\n%s", ledger.summary())

    ctx.add_shutdown_callback(log_ledger)

    await session.start(agent=LedgerAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
