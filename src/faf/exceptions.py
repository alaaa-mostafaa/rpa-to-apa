class DiagnostError(Exception):
    """Base exception for all diagnost errors."""
    pass

class AdapterAuthenticationError(DiagnostError):
    """Raised when an adapter fails to authenticate with its source system."""
    pass

class InsufficientMetadataError(DiagnostError):
    """Raised when an event lacks the required metadata for analysis."""
    pass

class CorpusNotFoundError(DiagnostError):
    """Raised when the retrieval corpus cannot be found or loaded."""
    pass

class MissingDependenciesError(DiagnostError):
    """Raised when optional dependencies for a specific adapter are missing."""
    pass
