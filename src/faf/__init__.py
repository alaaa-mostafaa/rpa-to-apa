from typing import Any, Dict

from faf.models import RunEvent, FailedStepInfo
from faf.agent import run_agent, AgentState
from faf.adapters.base import FailureAdapter
from faf.adapters.github import GitHubAdapter
from faf.adapters.kubernetes import KubernetesAdapter
from faf.bayesian_tracker import registry
from faf.exceptions import (
    DiagnostError, 
    AdapterAuthenticationError, 
    InsufficientMetadataError, 
    CorpusNotFoundError,
    MissingDependenciesError
)

class FailureAnalysisAgent:
    """
    Public entry point for the Diagnost package.
    Wraps the LangGraph loop and LLM initialization.
    """
    def __init__(self, **kwargs: Any):
        # In the future, LLM clients or API keys can be passed here.
        pass
        
    def analyze(self, event: RunEvent) -> Dict[str, Any]:
        """
        Analyzes a failure event and returns the predicted root cause category.
        """
        state = run_agent(event)
        return state["classification"]

__all__ = [
    "FailureAnalysisAgent",
    "RunEvent",
    "FailedStepInfo",
    "FailureAdapter",
    "GitHubAdapter",
    "KubernetesAdapter",
    "registry",
    "DiagnostError",
    "AdapterAuthenticationError",
    "InsufficientMetadataError",
    "CorpusNotFoundError",
    "MissingDependenciesError"
]
