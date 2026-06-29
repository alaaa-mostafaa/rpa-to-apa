import os
from dotenv import load_dotenv
from faf import FailureAnalysisAgent, GitHubAdapter, DiagnostError

load_dotenv()

def run_github_example():
    print("=== Diagnost GitHub Adapter Example ===")
    
    # In a real environment, you'd supply a GitHub Token
    token = os.environ.get("GITHUB_TOKEN")
    adapter = GitHubAdapter(token=token)
    
    # 1. Provide a known repository and run ID
    repo = "pyca/bcrypt"
    run_id = "12345678"  # Replace with a real failed run ID
    
    try:
        # Note: Without a valid token and run_id, this will raise an error.
        # This demonstrates our new clean exceptions!
        event = adapter.fetch_event(event_id=run_id, repo=repo)
    except DiagnostError as e:
        print(f"Caught expected DiagnostError during fetch: {e}")
        
        # Let's fallback to parsing a raw dictionary to demonstrate the agent
        print("\\nFalling back to parsing a raw run dictionary...")
        raw_run = {
            "_id": run_id,
            "repository_name": repo,
            "metadata": {
                "head_branch": "dependabot/pip/requests",
                "conclusion": "failure",
                "head_commit": {"message": "Bump requests to 2.31.0"}
            },
            "log_insights": [
                {
                    "file": "build.yml",
                    "steps": [
                        {"error": "ModuleNotFoundError: No module named 'requests'"}
                    ]
                }
            ]
        }
        event = adapter.parse_raw(raw_run)

    print(f"\\nSuccessfully loaded Event: {event.run_id} from {event.source}")
    print(f"Available Signals: {event.available_signals}")

    # 2. Initialize the Agent
    agent = FailureAnalysisAgent()
    
    # 3. Analyze the Failure
    print("\\nAnalyzing failure...")
    result = agent.analyze(event)
    
    print("\\n=== Diagnosis Result ===")
    print(f"Predicted Category: {result['category']}")
    print(f"Probability: {result['probability']:.1%}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"\\nReasoning Log:\\n{result['reasoning']}")

if __name__ == "__main__":
    run_github_example()
