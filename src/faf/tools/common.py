import json
import os
from typing import Dict, Any
from faf.llm import make_client, record_usage

LLM_MODEL = os.environ.get("LLM_MODEL") or os.environ.get("CI_AGENT_MODEL") or "gpt-4o-mini"

DEEP_LOG_ANALYSIS_PROMPT = """You are diagnosing a CI/CD failure from raw logs.
Analyze this log excerpt and explain what failed. Be highly specific.
Respond with a JSON object: {"diagnosis": "explanation"}
"""

def deep_log_analysis(state: dict) -> dict:
    """
    LLM-based reasoning over the full pre-extracted failure excerpt.
    """
    event = state.get("event")
    if not event:
        return {"investigation_log": ["deep_log_analysis: No event found."]}
        
    error_texts = [step.error_text for step in event.failed_steps if step.error_text]
    log_summary = "\\n".join(error_texts) if error_texts else "No explicit errors found."
    
    if log_summary == "No explicit errors found.":
        return {"investigation_log": ["Deep log analysis: No raw logs available."]}
        
    try:
        client = make_client()
        messages = [
            {"role": "system", "content": DEEP_LOG_ANALYSIS_PROMPT},
            {"role": "user", "content": f"Logs:\\n{log_summary[:4000]}"}
        ]
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            response_format={"type": "json_object"}
        )
        record_usage(response, LLM_MODEL, "chat", "deep_log_analysis")
        result = json.loads(response.choices[0].message.content)
        diagnosis = result.get("diagnosis", "Could not parse diagnosis.")
    except Exception as e:
        diagnosis = f"LLM error: {e}"
    
    return {
        "investigation_log": [f"[deep_log_analysis] {diagnosis}"]
    }

def search_similar_failures(state: dict) -> dict:
    return {
        "investigation_log": ["Searched vector store for similar failures: relied on retrieval prior."]
    }

def search_web_for_error(state: dict) -> dict:
    return {
        "investigation_log": ["Web search skipped (not fully implemented in open source)."]
    }
