from halo import gate_eval
from halo.randomized_tests import run_randomized_suite


def test_gate_agrees_with_oracle():
    result = run_randomized_suite(n_sequences=2000)
    assert result["total_release_attempts"] > 0
    assert result["oracle_permits"] > 0
    assert result["disagreements"] == []


def test_mutation_detection_pattern():
    m = gate_eval.mutation(n_sequences=2000)
    assert (m["live"], m["equivalent"]) == (14, 2)
    assert m["scenario_detects"] == 11
    assert m["randomized_detects"] == 12
    assert m["combined_detects"] == 13
    assert m["survive_both"] == ["M15_WeakenWindowExclDeadline"]
