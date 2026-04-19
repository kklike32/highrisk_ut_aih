from __future__ import annotations

# Re-export for `adk run` / `adk web` — implementation lives in `derm_advisor.adk_agent`
# so Streamlit and the CLI share one agent definition.
from derm_advisor.adk_agent import root_agent

__all__ = ["root_agent"]
