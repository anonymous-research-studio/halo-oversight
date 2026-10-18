import numpy as np

from halo.allocation import POLICIES, build, config
from halo.policies import LinearQLearner, LinUCBBandit, MinObligation, StaticGates
from halo.simulator import (
    RULE_HAZARD_COVERAGE, UNENCODED_HAZARDS, SimConfig, generate_workload, run_simulation,
)


def test_every_rule_can_fire():
    w = generate_workload(SimConfig(seed=1))
    for name, rule in RULE_HAZARD_COVERAGE.items():
        assert any(rule["feature_check"](c.features) for c in w.cases), name


def test_unencoded_hazards():
    assert UNENCODED_HAZARDS == frozenset({"outdated_guideline"})


def test_policies_face_identical_workloads():
    cfg = SimConfig(seed=7)
    a = run_simulation(MinObligation(), cfg)
    b = run_simulation(StaticGates(), cfg)
    assert (a.total_cases, a.hazardous_cases, a.obligated_cases) == \
           (b.total_cases, b.hazardous_cases, b.obligated_cases)


def test_coverage_changes_obligations_not_cases():
    full = run_simulation(MinObligation(), SimConfig(seed=7, rule_coverage=1.0))
    none = run_simulation(MinObligation(), SimConfig(seed=7, rule_coverage=0.0))
    assert none.obligated_cases == 0 < full.obligated_cases
    assert none.hazardous_cases == full.hazardous_cases


def test_gate_never_holds_a_simulated_release():
    for condition in ("baseline", "stress"):
        r = run_simulation(StaticGates(threshold=0.5), config(condition, seed=3))
        assert r.gate_holds == 0
        assert r.gate_permits == r.total_released == r.total_cases - r.total_withheld


def test_cost_decomposition():
    r = run_simulation(StaticGates(), SimConfig(seed=5))
    expected = r.harm_per_1000 + 0.02 * r.effort_per_1000 + r.withheld_per_1000
    assert abs(r.cost_per_1000 - expected) < 1e-9


def test_q_learner_is_deterministic():
    spec = next(p for p in POLICIES if p.key == "q_learner")
    costs = [run_simulation(build(spec, {}), SimConfig(seed=300)).cost_per_1000 for _ in range(2)]
    assert costs[0] == costs[1]


def test_frozen_learners_do_not_change():
    q = LinearQLearner(seed=0)
    run_simulation(q, SimConfig(seed=200))
    q.freeze()
    w = q.weights.copy()
    run_simulation(q, SimConfig(seed=300))
    assert np.array_equal(w, q.weights)

    b = LinUCBBandit()
    run_simulation(b, SimConfig(seed=200))
    b.freeze()
    A = [a.copy() for a in b.A]
    run_simulation(b, SimConfig(seed=300))
    assert all(np.array_equal(x, y) for x, y in zip(A, b.A))


def test_q_learner_parameter_count():
    assert LinearQLearner().n_params == 903
