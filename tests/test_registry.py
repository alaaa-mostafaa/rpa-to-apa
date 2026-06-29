import pytest
from faf import registry, RunEvent
from faf.bayesian_tracker import BeliefState

def test_registry_evaluation():
    @registry.register("test_signal")
    def dummy_signal(event: RunEvent):
        return {"CODE_REGRESSION": 0.8, "CONFIG_ERROR": 0.2}
        
    event = RunEvent(
        source="test",
        run_id="1",
        status="failure",
        failed_steps=[],
        available_signals=["test_signal"],
        metadata={}
    )
    
    results = registry.evaluate_available(event)
    assert "test_signal" in results
    assert results["test_signal"]["CODE_REGRESSION"] == 0.8

def test_belief_tracker_update():
    tracker = BeliefState()
    initial_confidence = tracker.confidence()
    
    tracker.update({"CODE_REGRESSION": 0.9, "CONFIG_ERROR": 0.1}, "test_signal")
    
    assert tracker.confidence() > initial_confidence
    top_cat, top_prob = tracker.top_category()
    assert top_cat == "CODE_REGRESSION"
