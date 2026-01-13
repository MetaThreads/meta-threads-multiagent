"""System prompts for all agents."""

from threads_multiagent.prompts.orchestrator import ORCHESTRATOR_PROMPT
from threads_multiagent.prompts.planning import PLANNING_PROMPT
from threads_multiagent.prompts.threads import THREADS_PROMPT

__all__ = [
    "PLANNING_PROMPT",
    "ORCHESTRATOR_PROMPT",
    "THREADS_PROMPT",
]
