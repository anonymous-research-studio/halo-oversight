from halo.allocation import POLICIES, breakeven_effort_weight, decision_rule


def _rows(costs_by_key):
    return {k: {"cost_mean": sum(v) / len(v), "per_seed_cost": v} for k, v in costs_by_key.items()}


def _table(learner_costs, static_costs):
    costs = {p.key: [200.0] * 3 for p in POLICIES}
    costs["static_gates"] = static_costs
    costs["linucb"] = learner_costs
    costs["q_learner"] = [300.0] * 3
    return {"baseline": _rows(costs)}


def test_decision_rule_needs_margin_and_every_seed():
    passes = decision_rule(_table([90.0, 90.0, 90.0], [100.0, 100.0, 100.0]))
    bandit = next(d for d in passes if d["learner"].startswith("LinUCB"))
    assert bandit["passes"]

    one_loss = decision_rule(_table([80.0, 80.0, 101.0], [100.0, 100.0, 100.0]))
    bandit = next(d for d in one_loss if d["learner"].startswith("LinUCB"))
    assert bandit["pct_lower"] > 5 and not bandit["passes"]

    small = decision_rule(_table([97.0, 97.0, 97.0], [100.0, 100.0, 100.0]))
    bandit = next(d for d in small if d["learner"].startswith("LinUCB"))
    assert bandit["lower_on_every_seed"] and not bandit["passes"]


def test_breakeven_weight():
    a = {"harm": 31.0, "effort": 1700.0, "withheld": 0.0}
    b = {"harm": 46.8, "effort": 1140.0, "withheld": 0.0}
    assert abs(breakeven_effort_weight(a, b) - 15.8 / 560) < 1e-12
