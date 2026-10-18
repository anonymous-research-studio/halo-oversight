"""Format results as text tables and write CSV files."""

import csv
import math
import os
from typing import Dict, List

from .allocation import CONDITION_LABELS, COVERAGES, POLICIES


def _f(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def header(title: str) -> str:
    return f"\n{'=' * 78}\n  {title}\n{'=' * 78}\n"


def format_table2(rows: List[Dict]) -> str:
    out = [header("Table 2: Scenario conformance (20 scenarios x 4 labels = 80 cases)")]
    out.append(f"{'Method':<26}{'Matched':>9}{'False permits':>16}{'False holds':>14}{'Unencoded released':>21}")
    out.append("-" * 86)
    for r in rows:
        out.append(f"{r['method']:<26}{r['matched']:>4}/{r['total']:<4}"
                   f"{r['false_permits']:>10}/{r['expected_holds']:<5}"
                   f"{r['false_holds']:>8}/{r['expected_permits']:<5}"
                   f"{r['unencoded_released']:>15}/{r['unencoded_total']:<5}")
    return "\n".join(out)


def format_randomized(r: Dict) -> str:
    return (header(f"Randomized suite: {r['sequences']:,} sequences vs. oracle")
            + f"Release attempts: {r['release_attempts']:,}\n"
            + f"Oracle permits:   {r['oracle_permits']:,}\n"
            + f"Disagreements:    {r['disagreements']}")


def format_mutation(m: Dict) -> str:
    out = [header(f"Mutation testing: {m['live'] + m['equivalent']} mutants "
                  f"({m['live']} live, {m['equivalent']} equivalent)")]
    out.append(f"{'Mutant':<34}{'Live':>6}{'Scenario':>10}{'Randomized':>12}{'Either':>8}")
    out.append("-" * 70)
    mark = lambda b: "yes" if b else "no"
    for r in m["rows"]:
        if r["live"]:
            out.append(f"{r['mutant']:<34}{'yes':>6}{mark(r['scenario']):>10}"
                       f"{mark(r['randomized']):>12}{mark(r['either']):>8}")
        else:
            out.append(f"{r['mutant']:<34}{'eq':>6}{mark(r['scenario']):>10}{'–':>12}{'–':>8}")
    out.append("")
    out.append(f"Scenario suite detects:   {m['scenario_detects']}/{m['live']} live mutants")
    out.append(f"Randomized suite detects: {m['randomized_detects']}/{m['live']} live mutants")
    out.append(f"Either suite detects:     {m['combined_detects']}/{m['live']} live mutants")
    out.append(f"Survive both suites:      {', '.join(m['survive_both']) or 'none'}")
    return "\n".join(out)


def format_tuning(tuning: List[Dict]) -> str:
    out = [header("Tuning: configuration chosen on validation seeds (baseline, coverage 1.0)")]
    labels = {p.key: p.label for p in POLICIES}
    for t in tuning:
        chosen = ", ".join(f"{k}={v}" for k, v in t["chosen"].items()) or "(no parameters)"
        best = min(c["validation_cost"] for c in t["candidates"])
        out.append(f"  {labels[t['policy']]:<27} {chosen:<45} val. cost {best:6.1f}"
                   f"  ({len(t['candidates'])} config{'s' if len(t['candidates']) != 1 else ''})")
    return "\n".join(out)


def format_table3(table3: Dict) -> str:
    out = [header("Table 3: Matched-policy comparison, 5 test seeds, full rule coverage"),
           "Cost, harm, effort, withheld per 1,000 arrivals; cost as mean ± sample s.d.\n"]
    cols = f"  {'Policy':<27}{'Cost':>13}{'Harm':>8}{'Effort':>8}{'With.':>7}{'Block%':>8}{'Any%':>6}{'Prec%':>7}{'Crit%':>7}"
    for cond, rows in table3.items():
        out.append(f"  {CONDITION_LABELS[cond]}")
        out.append(cols)
        out.append("  " + "-" * 91)
        for p in POLICIES:
            r = rows[p.key]
            cost = f"{r['cost_mean']:.1f}±{r['cost_sd']:.1f}"
            out.append(f"  {p.label:<27}{cost:>13}{_f(r['harm']):>8}{_f(r['effort'], 0):>8}"
                       f"{_f(r['withheld']):>7}{_f(r['block_pct'], 0):>8}{_f(r['any_pct'], 0):>6}"
                       f"{_f(r['disc_precision'], 0):>7}{_f(r['critical_catch_pct'], 0):>7}")
        out.append("")
    out.append("  Obligated cases by enforced level (%, pooled over test seeds): level 5 / 6 / 7")
    for cond, rows in table3.items():
        for p in POLICIES:
            lv = rows[p.key]["obligated_pct_level"]
            out.append(f"    [{cond}] {p.label:<27}{lv[5]:>6.0f}{lv[6]:>6.0f}{lv[7]:>6.0f}")
    return "\n".join(out)


def format_table4(table4: Dict) -> str:
    out = [header("Table 4: Harm per 1,000 arrivals as rule coverage falls (baseline)")]
    covs = "".join(f"{'Cov. ' + str(c):>11}" for c in COVERAGES)
    out.append(f"  {'Policy':<27}{covs}{'Unencoded haz. released %':>27}")
    out.append("  " + "-" * 87)
    for p in POLICIES:
        r = table4[p.key]
        vals = "".join(f"{_f(r['harm_by_coverage'][c]):>11}" for c in COVERAGES)
        out.append(f"  {p.label:<27}{vals}{_f(r['unencoded_released_pct'], 0):>27}")
    out.append("\n  Unencoded = hazard types no rule was written for "
               "(simulator.UNENCODED_HAZARDS), at coverage 1.0.")
    return "\n".join(out)


def format_decision(decision: List[Dict], title: str) -> str:
    out = [header(title)]
    for d in decision:
        verdict = "PASSES" if d["passes"] else "does not pass"
        diffs = ", ".join(f"{x:+.1f}" for x in d["per_seed_diff"])
        out.append(f"  [{d['condition']}] {d['learner']} vs {d['best_non_learner']}: "
                   f"{d['learner_cost']:.1f} vs {d['best_cost']:.1f} "
                   f"({d['pct_lower']:+.1f}% lower); per-seed diff [{diffs}] -> {verdict}")
    return "\n".join(out)


def write_csv(path: str, rows: List[Dict], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_gate_csvs(out_dir: str, t2: List[Dict], rand: Dict, mut: Dict) -> None:
    write_csv(os.path.join(out_dir, "gate", "table2_conformance.csv"), t2, list(t2[0].keys()))
    write_csv(os.path.join(out_dir, "gate", "randomized.csv"), [rand], list(rand.keys()))
    write_csv(os.path.join(out_dir, "gate", "mutation.csv"), mut["rows"], list(mut["rows"][0].keys()))


def write_allocation_csvs(out_dir: str, alloc: Dict) -> None:
    labels = {p.key: p.label for p in POLICIES}
    rows = []
    for cond, by_policy in alloc["table3"].items():
        for key, r in by_policy.items():
            row = {"condition": cond, "policy": labels[key]}
            row.update({k: v for k, v in r.items() if k not in ("per_seed_cost", "obligated_pct_level")})
            row.update({f"obligated_pct_level_{lvl}": v for lvl, v in r["obligated_pct_level"].items()})
            row.update({f"cost_seed_{i}": c for i, c in enumerate(r["per_seed_cost"])})
            rows.append(row)
    write_csv(os.path.join(out_dir, "allocation", "table3_policies.csv"), rows, list(rows[0].keys()))

    rows = []
    for key, r in alloc["table4"].items():
        row = {"policy": labels[key]}
        row.update({f"harm_cov_{c}": h for c, h in r["harm_by_coverage"].items()})
        row["unencoded_released_pct"] = r["unencoded_released_pct"]
        rows.append(row)
    write_csv(os.path.join(out_dir, "allocation", "table4_coverage.csv"), rows, list(rows[0].keys()))

    rows = []
    for t in alloc["tuning"]:
        for c in t["candidates"]:
            rows.append({"policy": labels[t["policy"]], "params": c["params"],
                         "validation_cost": c["validation_cost"],
                         "chosen": c["params"] == t["chosen"]})
    write_csv(os.path.join(out_dir, "allocation", "tuning.csv"), rows,
              ["policy", "params", "validation_cost", "chosen"])

    rows = []
    for tag, decisions in (("protocol", alloc["decision"]),
                           ("incl_post_hoc", alloc["decision_incl_post_hoc"])):
        for d in decisions:
            row = {"comparators": tag}
            row.update({k: v for k, v in d.items() if k != "per_seed_diff"})
            row["per_seed_diff"] = " ".join(f"{x:.2f}" for x in d["per_seed_diff"])
            rows.append(row)
    write_csv(os.path.join(out_dir, "allocation", "decision_rule.csv"), rows, list(rows[0].keys()))
