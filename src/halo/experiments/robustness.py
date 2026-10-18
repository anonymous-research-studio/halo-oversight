"""
Post hoc robustness analysis of the allocation study.

Not part of the preregistered protocol: these checks were added after the
protocol results were seen. Seeds 1000-1049 were never used for tuning or
training.

1. Fresh seeds: every protocol-tuned policy on 50 new test seeds, with paired
   cost differences against the best protocol-specified non-learning policy.
2. Strongest baseline: each rule-based policy is tuned on the 50 test seeds
   themselves over a finer grid (deliberately favoring the baselines), and
   the learners are compared with the best of them.
3. Q-learner seeds: the tuned Q-learner retrained with five exploration seeds.
4. Credit assignment: learners credited with the action they proposed rather
   than the action enforced after the rule floor, retuned and retrained.
5. Breakeven effort weight between the bandit and the strongest baseline.

    python -m halo robustness [--workers N]
"""

import math
import os
from multiprocessing import Pool
from typing import Dict, List, Optional

import numpy as np

from . import allocation as A
from ..simulation.policies import (
    TUNING_CONFIGS, CapacityAwareThreshold, ConfidenceThreshold, LinearQLearner,
    StaticGates, TwoThreshold,
)
from ..simulation.simulator import run_simulation

FRESH_SEEDS = list(range(1000, 1050))
Q_SEEDS = [0, 1, 2, 3, 4]

STRONGEST_BASELINE_GRIDS = {
    "static_gates": [{"threshold": round(t, 2)} for t in np.arange(0.4, 1.41, 0.05)],
    "conf_threshold": [{"threshold": round(t, 2)} for t in np.arange(0.2, 0.96, 0.05)],
    "cap_aware": [{"base_threshold": round(b, 2), "queue_sensitivity": q}
                  for b in np.arange(0.4, 0.96, 0.05) for q in (0.0, 0.01, 0.03, 0.05)],
    "two_threshold": [{"monitor_threshold": round(m, 2), "approval_threshold": round(a, 2)}
                      for m in np.arange(0.5, 0.91, 0.1)
                      for a in np.arange(0.2, 0.61, 0.1) if a < m - 1e-9],
}
_CLASSES = {"static_gates": StaticGates, "conf_threshold": ConfidenceThreshold,
            "cap_aware": CapacityAwareThreshold, "two_threshold": TwoThreshold}


class ProposedCredit:
    """Wraps a learner so updates credit the action it proposed, not the
    action enforced after the rule floor."""
    sees_queue = False
    learns = True

    def __init__(self, inner):
        self.inner = inner
        self.last = None

    def select(self, features, queue_len=None):
        self.last = self.inner.select(features, queue_len)
        return self.last

    def update(self, features, action, reward):
        self.inner.update(features, self.last, reward)

    def end_episode(self):
        self.inner.end_episode()

    def freeze(self):
        self.inner.freeze()


def _spec(key):
    return next(p for p in A.POLICIES if p.key == key)


def build_policy(key: str, params: dict, credit: str = "enforced"):
    """Build a policy; learners are trained on the training seeds and frozen."""
    spec = _spec(key)
    if credit == "enforced":
        return A.build(spec, params)
    kwargs = dict(params)
    if spec.cls is LinearQLearner:
        kwargs.setdefault("seed", A.POLICY_SEED)
    policy = ProposedCredit(spec.cls(**kwargs))
    for seed in A.TRAIN_SEEDS:
        run_simulation(policy, A.config("baseline", seed))
    policy.freeze()
    return policy


def _evaluate(job):
    key, params, credit, condition, seeds = job
    if credit == "rule":
        policy = _CLASSES[key](**params)
    else:
        policy = build_policy(key, params, credit)
    rs = [run_simulation(policy, A.config(condition, s)) for s in seeds]
    return {
        "cost": [r.cost_per_1000 for r in rs],
        "harm": [r.harm_per_1000 for r in rs],
        "effort": [r.effort_per_1000 for r in rs],
        "withheld": [r.withheld_per_1000 for r in rs],
    }


def _validation_cost(job):
    key, params, credit = job
    policy = build_policy(key, params, credit)
    return float(np.mean([run_simulation(policy, A.config("baseline", s)).cost_per_1000
                          for s in A.VALIDATION_SEEDS]))


def paired(a: List[float], b: List[float]) -> Dict:
    """Mean of a - b with a normal-approximation 95% CI, and seeds where a < b."""
    d = np.asarray(a) - np.asarray(b)
    half = 1.96 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {"diff": float(d.mean()), "ci_low": float(d.mean() - half),
            "ci_high": float(d.mean() + half),
            "pct_lower": float(-100.0 * d.mean() / np.mean(b)),
            "wins": int((d < 0).sum()), "n": int(len(d))}


def run(seeds: Optional[List[int]] = None, q_seeds: Optional[List[int]] = None,
        grids: Optional[Dict] = None, workers: Optional[int] = None,
        progress=print) -> Dict:
    seeds = seeds or FRESH_SEEDS
    q_seeds = q_seeds if q_seeds is not None else Q_SEEDS
    grids = grids or STRONGEST_BASELINE_GRIDS
    workers = workers or os.cpu_count() or 1
    pool = Pool(workers) if workers > 1 else None
    pmap = pool.map if pool else lambda f, xs: list(map(f, xs))

    try:
        progress("  tuning (protocol) ...")
        chosen = {spec.key: A.tune(spec)["chosen"] for spec in A.POLICIES}

        progress("  tuning learners with proposed-action credit ...")
        chosen_proposed = {}
        for key in ("linucb", "q_learner"):
            cands = TUNING_CONFIGS[key]
            costs = pmap(_validation_cost, [(key, p, "proposed") for p in cands])
            chosen_proposed[key] = cands[int(np.argmin(costs))]

        out = {"seeds": seeds, "chosen": chosen, "chosen_proposed": chosen_proposed,
               "conditions": {}}
        for condition in A.CONDITIONS:
            progress(f"  evaluating {condition} on {len(seeds)} seeds ...")
            # 1. Protocol-tuned policies on fresh seeds.
            jobs = [(s.key, chosen[s.key], "enforced", condition, seeds) for s in A.POLICIES]
            protocol = dict(zip([s.key for s in A.POLICIES], pmap(_evaluate, jobs)))

            # 2. Strongest baseline: tuned on the test seeds themselves.
            strongest = {}
            for key, grid in grids.items():
                res = pmap(_evaluate, [(key, p, "rule", condition, seeds) for p in grid])
                i = int(np.argmin([np.mean(r["cost"]) for r in res]))
                strongest[key] = {"params": grid[i], "result": res[i], "n_configs": len(grid)}

            # 3. Q-learner seeds, 4. proposed-action credit.
            q_runs = pmap(_evaluate, [("q_learner", dict(chosen["q_learner"], seed=s),
                                       "enforced", condition, seeds) for s in q_seeds])
            proposed = dict(zip(("linucb", "q_learner"), pmap(_evaluate, [
                (k, chosen_proposed[k], "proposed", condition, seeds) for k in ("linucb", "q_learner")])))

            best_protocol = min((s.key for s in A.POLICIES if not s.learns and s.protocol_specified),
                                key=lambda k: np.mean(protocol[k]["cost"]))
            best_any = min(strongest, key=lambda k: np.mean(strongest[k]["result"]["cost"]))
            ref_protocol = protocol[best_protocol]
            ref_strong = strongest[best_any]["result"]

            learners = [("LinUCB bandit", "enforced", protocol["linucb"])]
            learners += [(f"Q-learner, seed {s}" + (" (protocol)" if s == A.POLICY_SEED else ""),
                          "enforced", protocol["q_learner"] if s == A.POLICY_SEED else r)
                         for s, r in zip(q_seeds, q_runs)]
            learners += [("LinUCB bandit", "proposed", proposed["linucb"]),
                         ("Q-learner, seed 0", "proposed", proposed["q_learner"])]

            # Compare with the strongest baseline separately only if it differs
            # from the protocol-tuned best.
            same = (best_any == best_protocol
                    and strongest[best_any]["params"] == chosen[best_protocol])
            refs = [("protocol", ref_protocol)] if same else \
                   [("protocol", ref_protocol), ("strongest", ref_strong)]

            comparisons = []
            for name, credit, r in learners:
                for ref_name, ref in refs:
                    row = {"learner": name, "credit": credit, "baseline": ref_name,
                           "learner_cost": float(np.mean(r["cost"])),
                           "baseline_cost": float(np.mean(ref["cost"]))}
                    row.update(paired(r["cost"], ref["cost"]))
                    comparisons.append(row)

            cond_out = {"protocol": protocol, "strongest": strongest,
                        "best_protocol": best_protocol, "best_strongest": best_any,
                        "strongest_is_protocol": same, "comparisons": comparisons}
            b = protocol["linucb"]
            d_harm = (np.mean(ref_strong["harm"]) + np.mean(ref_strong["withheld"])) - \
                     (np.mean(b["harm"]) + np.mean(b["withheld"]))
            d_effort = np.mean(b["effort"]) - np.mean(ref_strong["effort"])
            cond_out["breakeven_effort_weight"] = float(d_harm / d_effort) if d_effort else None
            out["conditions"][condition] = cond_out
        return out
    finally:
        if pool:
            pool.close()
            pool.join()


def _mean(r, k):
    return float(np.mean(r[k]))


def format_report(res: Dict) -> str:
    labels = {p.key: p.label for p in A.POLICIES}
    n = len(res["seeds"])
    lines = ["\n" + "=" * 78,
             f"  Post hoc robustness analysis ({n} fresh test seeds, "
             f"{res['seeds'][0]}-{res['seeds'][-1]}; not preregistered)",
             "=" * 78 + "\n"]
    for condition, c in res["conditions"].items():
        lines.append(f"  {A.CONDITION_LABELS[condition]}")
        lines.append(f"  {'Policy (protocol-tuned)':<27}{'Cost':>8}{'Harm':>8}{'Effort':>8}{'With.':>7}")
        lines.append("  " + "-" * 58)
        for key, r in c["protocol"].items():
            lines.append(f"  {labels[key]:<27}{_mean(r, 'cost'):>8.1f}{_mean(r, 'harm'):>8.1f}"
                         f"{_mean(r, 'effort'):>8.0f}{_mean(r, 'withheld'):>7.1f}")
        lines.append("")
        lines.append("  Strongest baselines (tuned on these test seeds):")
        for key, s in c["strongest"].items():
            params = ", ".join(f"{k}={v}" for k, v in s["params"].items())
            lines.append(f"    {labels[key]:<27}{params:<46} cost {_mean(s['result'], 'cost'):6.1f}"
                         f"  ({s['n_configs']} configs)")
        if c["strongest_is_protocol"]:
            lines.append(f"  The strongest baseline is the protocol-tuned "
                         f"{labels[c['best_protocol']]} itself, so learners are compared with it once.")
        else:
            lines.append(f"  Best protocol-tuned non-learner: {labels[c['best_protocol']]}; "
                         f"strongest baseline: {labels[c['best_strongest']]}")
        lines.append("")
        lines.append(f"  {'Learner':<29}{'Credit':<10}{'vs':<11}{'Cost diff [95% CI]':>22}{'% lower':>9}{'Wins':>8}")
        lines.append("  " + "-" * 89)
        for r in c["comparisons"]:
            ci = f"{r['diff']:+.1f} [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}]"
            lines.append(f"  {r['learner']:<29}{r['credit']:<10}{r['baseline']:<11}{ci:>22}"
                         f"{r['pct_lower']:>+9.1f}{r['wins']:>5}/{r['n']}")
        lines.append("  Credit: 'enforced' = learner credited with the action after the rule floor\n"
                     "  (as in the protocol); 'proposed' = credited with its own proposal.")
        if condition == "baseline" and c["breakeven_effort_weight"] is not None:
            lines.append(f"\n  Breakeven effort weight, bandit vs strongest baseline: "
                         f"{c['breakeven_effort_weight']:.3f} per reviewer minute (declared 0.02)")
        lines.append("")
    return "\n".join(lines)


def write_csvs(out_dir: str, res: Dict) -> None:
    from .report import write_csv
    labels = {p.key: p.label for p in A.POLICIES}
    rows = []
    for condition, c in res["conditions"].items():
        for key, r in c["protocol"].items():
            rows.append({"condition": condition, "analysis": "protocol-tuned", "policy": labels[key],
                         "params": res["chosen"][key], "cost": _mean(r, "cost"),
                         "harm": _mean(r, "harm"), "effort": _mean(r, "effort"),
                         "withheld": _mean(r, "withheld")})
        for key, s in c["strongest"].items():
            r = s["result"]
            rows.append({"condition": condition, "analysis": "strongest (tuned on test seeds)",
                         "policy": labels[key], "params": s["params"], "cost": _mean(r, "cost"),
                         "harm": _mean(r, "harm"), "effort": _mean(r, "effort"),
                         "withheld": _mean(r, "withheld")})
    write_csv(os.path.join(out_dir, "robustness", "policies.csv"), rows, list(rows[0].keys()))
    rows = []
    for condition, c in res["conditions"].items():
        for r in c["comparisons"]:
            rows.append(dict(condition=condition, **r))
    write_csv(os.path.join(out_dir, "robustness", "comparisons.csv"), rows, list(rows[0].keys()))
