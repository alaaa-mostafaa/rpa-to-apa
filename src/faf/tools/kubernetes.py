from typing import Dict, Any

def inspect_k8s_events(state: dict) -> dict:
    """Fetches recent Kubernetes warning events for the pod/deployment."""
    event = state.get("event")
    events = event.metadata.get("k8s_events", []) if event else []
    
    if events:
        log_msg = "Found Kubernetes warnings:\\n" + "\\n".join(events)
    else:
        log_msg = "No Kubernetes warning events found."
        
    return {
        "investigation_log": [f"Inspected Kubernetes events. {log_msg}"]
    }
