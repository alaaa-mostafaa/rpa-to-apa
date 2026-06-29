# Contributing to Diagnost

We welcome new `FailureAdapter` implementations!

## Building a New Adapter

To add support for a new CI/CD provider (like GitLab or CircleCI), simply inherit from `FailureAdapter` and construct a `RunEvent`:

```python
from faf import FailureAdapter, RunEvent, FailedStepInfo

class GitLabAdapter(FailureAdapter):
    @property
    def source_name(self) -> str:
        return "gitlab"

    @property
    def available_signals(self) -> list[str]:
        return ["error_text", "branch_type"]

    def fetch_event(self, event_id: str, **kwargs) -> RunEvent:
        # Fetch from GitLab API and return a RunEvent
        ...
```

## Registering Signals

You can register custom Bayesian signals directly into the global `registry` if your adapter exposes unique metadata:

```python
from faf import registry

@registry.register("gitlab_pipeline_type")
def gitlab_pipeline_type(event: RunEvent) -> dict[str, float]:
    # Return a dictionary of category probabilities based on the event metadata
    ...
```
