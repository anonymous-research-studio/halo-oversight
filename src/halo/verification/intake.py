"""
Intake for externally generated scenario files (Section 2.4).

A scenario file is JSON: either a list of scenarios or {"scenarios": [...]},
each in the schema documented in halo/verification/scenarios.py. The "expected" field is
optional; generated scenarios usually arrive without one.

Each scenario is run through the gate, the oracle and the static checker.
The intake only reports: it never changes rules, contracts or the gate.
Generation, adjudication and testing stay separate.

    halo intake path/to/scenarios.json [more.json ...]
"""

import json
from typing import Dict, List, Optional, Tuple

from . import checker
from .oracle import Oracle, OracleContract, _hash as oracle_hash
from .scenarios import run_on_gate

VALID_EVENTS = {"review", "register_reviewer", "revoke_reviewer", "release"}


def validate(spec: dict) -> List[str]:
    """Return a list of problems with a scenario (empty if valid)."""
    problems = []
    if "events" not in spec or not isinstance(spec["events"], list):
        return ["missing 'events' list"]
    if not any(ev.get("type") == "release" for ev in spec["events"]):
        problems.append("no release event")
    for i, ev in enumerate(spec["events"]):
        if ev.get("type") not in VALID_EVENTS:
            problems.append(f"event {i}: unknown type {ev.get('type')!r}")
        if ev.get("type") == "review" and ev.get("decision") not in ("approve", "reject"):
            problems.append(f"event {i}: decision must be 'approve' or 'reject'")
    c = spec.get("contract")
    if c is not None:
        for key in ("output", "rule_revision", "obs_revision", "required_roles",
                    "min_reviewers", "floor", "window_start", "deadline"):
            if key not in c:
                problems.append(f"contract missing {key!r}")
    return problems


def run_on_oracle(spec: dict, case_id: str) -> Tuple[bool, Optional[str]]:
    oracle = Oracle()
    for rid, roles in spec.get("reviewers", {}).items():
        oracle.register(rid, set(roles))
    c = spec.get("contract")
    if c is not None:
        oracle.issue(OracleContract(
            case_id, oracle_hash(c["output"]), c["rule_revision"], c["obs_revision"],
            frozenset(c["required_roles"]), int(c["min_reviewers"]), int(c["floor"]),
            float(c["window_start"]), float(c["deadline"]),
        ))
    result: Tuple[bool, Optional[str]] = (False, None)
    for ev in spec["events"]:
        kind = ev["type"]
        if kind == "review":
            oracle.review(case_id, ev["reviewer"], ev["decision"], float(ev["time"]))
        elif kind == "register_reviewer":
            oracle.register(ev["reviewer"], set(ev["roles"]))
        elif kind == "revoke_reviewer":
            oracle.unregister(ev["reviewer"])
        elif kind == "release":
            result = oracle.check_release(case_id, oracle_hash(ev["output"]),
                                          ev["rule_revision"], ev["obs_revision"],
                                          float(ev["time"]), int(ev["action"]))
    return result


def load(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("scenarios", [])
    return list(data)


def evaluate(specs: List[dict]) -> List[Dict]:
    rows = []
    for i, spec in enumerate(specs):
        name = spec.get("name", f"scenario_{i}")
        problems = validate(spec)
        if problems:
            rows.append({"name": name, "valid": False, "problems": problems})
            continue
        case_id = f"intake_{i:04d}"
        gate = run_on_gate(spec, case_id)
        oracle = run_on_oracle(spec, case_id)
        static = checker.check(spec)
        expected = spec.get("expected")
        rows.append({
            "name": name, "valid": True,
            "gate": gate, "oracle": oracle, "checker": static,
            "expected": (expected["permit"], expected.get("reason")) if expected else None,
            "agree": gate[0] == oracle[0] == static[0],
            "matches_expected": None if not expected else gate[0] == expected["permit"],
        })
    return rows


def format_report(rows: List[Dict]) -> str:
    def fmt(r):
        return "permit" if r[0] else f"hold({r[1]})"
    lines = []
    for r in rows:
        if not r["valid"]:
            lines.append(f"{r['name']}: INVALID: {'; '.join(r['problems'])}")
            continue
        flag = "" if r["agree"] else "  <-- gate/oracle/checker DISAGREE"
        exp = ""
        if r["expected"] is not None:
            exp = f"  expected={fmt(r['expected'])}" + ("" if r["matches_expected"] else "  <-- MISMATCH")
        lines.append(f"{r['name']}: gate={fmt(r['gate'])} oracle={fmt(r['oracle'])} "
                     f"checker={fmt(r['checker'])}{exp}{flag}")
    n_valid = sum(r["valid"] for r in rows)
    n_dis = sum(1 for r in rows if r["valid"] and not r["agree"])
    lines.append(f"\n{len(rows)} scenarios, {n_valid} valid, {n_dis} with disagreement")
    return "\n".join(lines)
