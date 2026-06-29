import os
from dotenv import load_dotenv
from faf import FailureAnalysisAgent
from faf.adapters.kubernetes import KubernetesAdapter

load_dotenv()

def run_k8s_example():
    print("=== Diagnost Kubernetes Adapter Example ===")
    
    adapter = KubernetesAdapter()
    
    # Let's mock a payload for a failing pod
    payload = {
        "namespace": "default",
        "pod_name": "payment-service-123",
        "phase": "Failed",
        "reason": "OOMKilled",
        "logs": "Memory limit exceeded...",
        "events": [
            "Warning: Memory cgroup out of memory: Killed process 1234 (python)"
        ],
        "labels": {"commit_sha": "abc1234"}
    }
    
    print("\\nParsing mocked Kubernetes OOM_KILL pod payload...")
    event = adapter.parse_raw(payload)
    
    print(f"Successfully loaded Event: {event.run_id} from {event.source}")
    print(f"Available Signals: {event.available_signals}")

    agent = FailureAnalysisAgent()
    
    print("\\nAnalyzing failure...")
    result = agent.analyze(event)
    
    print("\\n=== Diagnosis Result ===")
    print(f"Predicted Category: {result['category']}")
    print(f"Probability: {result['probability']:.1%}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"\\nReasoning Log:\\n{result['reasoning']}")

if __name__ == "__main__":
    run_k8s_example()
