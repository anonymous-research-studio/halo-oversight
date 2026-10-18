"""
Allocation study (Sections 2.3, 3.3 to 3.5).

Protocol, as implemented here:
1. Tuning: each tunable policy tries at most eight configurations
   (policies.TUNING_CONFIGS) and keeps the one with the lowest mean cost on
   the three validation seeds, baseline condition, full rule coverage.
2. Learners (LinUCB, Q-learner) are trained on the three training seeds in
   sequence, then frozen. Learner configurations are tuned the same way:
   train, freeze, score on the validation seeds.
3. Every policy is evaluated on the five test seeds. The capacity-stress
   condition is never used for tuning or training.
4. Decision rule: a learner counts as better than the best protocol-specified
   non-learning policy only if its mean cost is more than 5% lower and its
   paired cost difference is negative on every test seed.

The simulator draws all randomness for a run up front (simulator.Workload),
so every policy faces identical cases on a given seed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..simulation.policies import (
    TUNING_CONFIGS, CapacityAwareThreshold, ConfidenceThreshold, LinearQLearner,
    LinUCBBandit, MinObligation, RiskMatrix, StaticGates, TwoThreshold,
)
from ..simulation.simulator import SimConfig, SimResults, run_simulation

VALIDATION_SEEDS = [100, 101, 102]
TRAIN_SEEDS = [200, 201, 202]
TEST_SEEDS = [300, 301, 302, 303, 304]
POLICY_SEED = 0          # seeds the Q-learner's exploration
COVERAGES = [1.0, 0.5, 0.0]
DECISION_MARGIN_PCT = 5.0

CONDITIONS = {
    "baseline": {"arrival_rate": 1.0, "n_generalists": 4, "n_specialists": 2},
    "stress": {"arrival_rate": 1.6, "n_generalists": 2, "n_specialists": 1},
}
CONDITION_LABELS = {
    "baseline": "Baseline: 1.0 arrivals/min, 4 generalists + 2 specialists",
    "stress": "Capacity stress (held out): 1.6 arrivals/min, 2 generalists + 1 specialist",
}


@dataclass(frozen=True)
class PolicySpec:
    key: str
    label: str
    cls: type
    protocol_specified: bool = True

    @property
    def learns(self) -> bool:
        return getattr(self.cls, "learns", False)


POLICIES = [
    PolicySpec("min_obligation", "Min. obligation", MinObligation),
    PolicySpec("static_gates", "Static gates (B2)", StaticGates),
    PolicySpec("conf_threshold", "Conf. threshold (B3)", ConfidenceThreshold),
    PolicySpec("risk_matrix", "Risk matrix (B4)", RiskMatrix),
    PolicySpec("cap_aware", "Cap.-aware threshold†", CapacityAwareThreshold),
    PolicySpec("two_threshold", "Two-threshold (post hoc)", TwoThreshold, protocol_specified=False),
    PolicySpec("linucb", "LinUCB bandit (B5)", LinUCBBandit),
    PolicySpec("q_learner", "Ref. linear Q-learner", LinearQLearner),
]


def config(condition: str = "baseline", seed: int = 0, coverage: float = 1.0) -> SimConfig:
    return SimConfig(seed=seed, rule_coverage=coverage, **CONDITIONS[condition])


def build(spec: PolicySpec, params: dict):
    """Instantiate a policy; learners are trained on TRAIN_SEEDS and frozen."""
    kwargs = dict(params)
    if spec.cls is LinearQLearner:
        kwargs.setdefault("seed", POLICY_SEED)
    policy = spec.cls(**kwargs)
    if spec.learns:
        for seed in TRAIN_SEEDS:
            run_simulation(policy, config("baseline", seed))
        policy.freeze()
    return policy


def evaluate(policy, seeds: List[int], condition: str = "baseline",
             coverage: float = 1.0) -> List[SimResults]:
    return [run_simulation(policy, config(condition, s, coverage)) for s in seeds]


def tune(spec: PolicySpec) -> Dict:
    """Pick the configuration with the lowest mean validation cost."""
    candidates = TUNING_CONFIGS.get(spec.key, [{}])
    scored = []
    for params in candidates:
        policy = build(spec, params)
        cost = float(np.mean([r.cost_per_1000 for r in evaluate(policy, VALIDATION_SEEDS)]))
        scored.append({"params": params, "validation_cost": cost})
    best = min(range(len(scored)), key=lambda i: scored[i]["validation_cost"])
    return {"policy": spec.key, "chosen": scored[best]["params"], "candidates": scored}


def summarize(results: List[SimResults]) -> Dict:
    costs = [r.cost_per_1000 for r in results]
    prec = [r.disc_precision for r in results if not np.isnan(r.disc_precision)]
    return {
        "cost_mean": float(np.mean(costs)),
        "cost_sd": float(np.std(costs, ddof=1)),
        "harm": float(np.mean([r.harm_per_1000 for r in results])),
        "effort": float(np.mean([r.effort_per_1000 for r in results])),
        "withheld": float(np.mean([r.withheld_per_1000 for r in results])),
        "block_pct": float(np.mean([r.block_pct for r in results])),
        "any_pct": float(np.mean([r.any_pct for r in results])),
        "disc_precision": float(np.mean(prec)) if prec else float("nan"),
        "critical_catch_pct": float(np.mean([r.critical_catch_pct for r in results])),
        "unencoded_released_pct": float(np.nanmean([r.unencoded_released_pct for r in results])),
        "gate_holds": int(sum(r.gate_holds for r in results)),
        "obligated_pct_level": _obligated_levels(results),
        "per_seed_cost": costs,
    }


def _obligated_levels(results: List[SimResults]) -> Dict[int, float]:
    """Share of obligated cases (pooled over seeds) assigned each enforced level."""
    counts: Dict[int, int] = {}
    for r in results:
        for level, n in r.obligated_by_action.items():
            counts[level] = counts.get(level, 0) + n
    total = sum(counts.values())
    return {lvl: 100.0 * counts.get(lvl, 0) / total if total else 0.0 for lvl in (5, 6, 7)}


def decision_rule(table3: Dict[str, Dict[str, Dict]], include_post_hoc: bool = False) -> List[Dict]:
    """Apply the protocol's decision rule to each learner in each condition."""
    out = []
    for condition, rows in table3.items():
        comparators = [s for s in POLICIES if not s.learns
                       and (s.protocol_specified or include_post_hoc)]
        best = min(comparators, key=lambda s: rows[s.key]["cost_mean"])
        best_row = rows[best.key]
        for spec in POLICIES:
            if not spec.learns:
                continue
            row = rows[spec.key]
            diffs = [a - b for a, b in zip(row["per_seed_cost"], best_row["per_seed_cost"])]
            pct_lower = 100.0 * (best_row["cost_mean"] - row["cost_mean"]) / best_row["cost_mean"]
            out.append({
                "condition": condition,
                "learner": spec.label,
                "best_non_learner": best.label,
                "learner_cost": row["cost_mean"],
                "best_cost": best_row["cost_mean"],
                "pct_lower": pct_lower,
                "per_seed_diff": diffs,
                "lower_on_every_seed": all(d < 0 for d in diffs),
                "passes": pct_lower > DECISION_MARGIN_PCT and all(d < 0 for d in diffs),
            })
    return out


def breakeven_effort_weight(row_a: Dict, row_b: Dict) -> Optional[float]:
    """Effort weight at which two policies' costs are equal (withheld held at 1.0)."""
    d_effort = row_a["effort"] - row_b["effort"]
    if d_effort == 0:
        return None
    return ((row_b["harm"] + row_b["withheld"]) - (row_a["harm"] + row_a["withheld"])) / d_effort


def run(progress=print) -> Dict:
    """Run the full allocation study and return all results."""
    tuning, policies = [], {}
    for spec in POLICIES:
        progress(f"  tuning/training {spec.label} ...")
        t = tune(spec)
        tuning.append(t)
        policies[spec.key] = build(spec, t["chosen"])

    table3 = {}
    for condition in CONDITIONS:
        table3[condition] = {spec.key: summarize(evaluate(policies[spec.key], TEST_SEEDS, condition))
                             for spec in POLICIES}

    table4 = {}
    for spec in POLICIES:
        harms = {}
        for cov in COVERAGES:
            res = evaluate(policies[spec.key], TEST_SEEDS, "baseline", cov)
            harms[cov] = float(np.mean([r.harm_per_1000 for r in res]))
        table4[spec.key] = {"harm_by_coverage": harms,
                            "unencoded_released_pct": table3["baseline"][spec.key]["unencoded_released_pct"]}

    return {
        "tuning": tuning,
        "table3": table3,
        "table4": table4,
        "decision": decision_rule(table3),
        "decision_incl_post_hoc": decision_rule(table3, include_post_hoc=True),
        "q_params": LinearQLearner().n_params,
    }
