from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class FailedStepInfo:
    """Information about a specific failed step in a run."""
    name: str
    error_text: str
    exit_code: Optional[int] = None
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RunEvent:
    """A generalized run event representing a failure from any source."""
    source: str  # e.g., 'github', 'kubernetes'
    run_id: str
    status: str
    url: Optional[str] = None
    failed_steps: List[FailedStepInfo] = field(default_factory=list)
    available_signals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the RunEvent to a dictionary for LLM context or logging."""
        return {
            "source": self.source,
            "run_id": self.run_id,
            "status": self.status,
            "url": self.url,
            "failed_steps": [
                {
                    "name": step.name,
                    "error_text": step.error_text[:2000] + "..." if len(step.error_text) > 2000 else step.error_text,
                    "exit_code": step.exit_code,
                    "duration_seconds": step.duration_seconds,
                    "metadata": step.metadata
                } for step in self.failed_steps
            ],
            "available_signals": self.available_signals,
            "metadata": self.metadata
        }
