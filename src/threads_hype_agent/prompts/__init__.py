"""System prompts for all agents."""

from threads_hype_agent.prompts.orchestrator import ORCHESTRATOR_PROMPT
from threads_hype_agent.prompts.planning import PLANNING_PROMPT
from threads_hype_agent.prompts.threads import THREADS_PROMPT

__all__ = [
    "PLANNING_PROMPT",
    "ORCHESTRATOR_PROMPT",
    "THREADS_PROMPT",
]
