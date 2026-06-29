from abc import ABC, abstractmethod
from typing import List, Any
from faf.models import RunEvent

class FailureAdapter(ABC):
    """
    Base class for all failure adapters.
    An adapter is responsible for fetching failure data from a specific source (e.g., GitHub, Kubernetes)
    and returning a generalized RunEvent.
    """
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """The name of the source system (e.g., 'github', 'kubernetes')."""
        pass

    @property
    @abstractmethod
    def available_signals(self) -> List[str]:
        """A list of signals that this adapter typically provides metadata for."""
        pass

    @abstractmethod
    def fetch_event(self, event_id: str, **kwargs: Any) -> RunEvent:
        """
        Fetch the failure details and return a standardized RunEvent.
        
        Args:
            event_id: The unique identifier for the event (e.g., GitHub run ID, Kubernetes pod name).
            **kwargs: Additional source-specific configuration.
        """
        pass
