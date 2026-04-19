from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.runners import Runner


def _text_from_content(content: Any) -> str:
    if content is None:
        return ""
    parts = getattr(content, "parts", None) or []
    out: list[str] = []
    for p in parts:
        t = getattr(p, "text", None)
        if t:
            out.append(str(t))
    return "\n".join(out)


def _collect_visible_reply_text(events: list[Any]) -> str:
    """Join assistant/model text turns, skipping tool call/response-only events when possible."""
    chunks: list[str] = []
    for e in events:
        try:
            if hasattr(e, "get_function_calls") and e.get_function_calls():
                continue
            if hasattr(e, "get_function_responses") and e.get_function_responses():
                continue
        except Exception:
            pass
        auth = getattr(e, "author", "") or ""
        if auth == "user":
            continue
        t = _text_from_content(getattr(e, "content", None))
        if t.strip():
            chunks.append(t.strip())
    return "\n\n".join(chunks).strip()


def create_runner(agent: "BaseAgent") -> "Runner":
    """
    Build a Runner backed by an in-memory session store — same stack `adk web` uses for local dev.
    Hold the returned Runner across turns (e.g. in Streamlit session_state) so conversation persists.
    """
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService

    return Runner(
        app_name="derm_advisor_app",
        agent=agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )


def run_turn(*, runner: "Runner", user_id: str, session_id: str, text: str) -> str:
    """Run one user text turn and return assistant-visible text."""
    from google.genai import types

    msg = types.Content(role="user", parts=[types.Part(text=text)])
    events = list(
        runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=msg,
        )
    )
    return _collect_visible_reply_text(events)
