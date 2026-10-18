"""
Static contract checker and ordinal-only control for the scenario suite.

``check`` evaluates the release condition of Section 2.1 directly from a
scenario's declared fields. It shares no code with gate.py or oracle.py:
outputs are compared as strings rather than hashes, and every condition is
evaluated before the first failing one (in the gate's reporting order) is
returned.

``ordinal_only`` is the control of Table 2: it checks only the floor of
Equation 1 (enforced action >= floor) and nothing else.
"""

from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

# Reporting order, matching the gate's reason codes.
REASON_ORDER = [
    "unknown_case",
    "already_released",
    "payload_changed",
    "rules_changed",
    "observations_changed",
    "outside_release_window",
    "action_below_floor",
    "qualified_rejection",
    "missing_role_coverage",
    "too_few_distinct_reviewers",
]


class _Review(NamedTuple):
    reviewer: str
    roles_then: FrozenSet[str]
    decision: str
    time: float


def _failures(contract: Optional[dict], released_before: bool,
              reviews: List[_Review], registry: Dict[str, FrozenSet[str]],
              ev: dict) -> Dict[str, bool]:
    if contract is None:
        return {"unknown_case": True}
    t = float(ev["time"])
    required = frozenset(contract["required_roles"])
    t0, d = float(contract["window_start"]), float(contract["deadline"])

    # Reviews are recorded against the contract's own output and revisions,
    # so applicability depends only on the review time.
    applicable = [r for r in reviews if t0 <= r.time <= min(t, d)]
    counted = [r for r in applicable if r.reviewer in registry and r.roles_then & required]
    approvals = [r for r in counted if r.decision == "approve"]
    covered = frozenset().union(*(r.roles_then & required for r in approvals))

    return {
        "already_released": released_before,
        "payload_changed": ev["output"] != contract["output"],
        "rules_changed": ev["rule_revision"] != contract["rule_revision"],
        "observations_changed": ev["obs_revision"] != contract["obs_revision"],
        "outside_release_window": not (t0 <= t <= d),
        "action_below_floor": int(ev["action"]) < int(contract["floor"]),
        "qualified_rejection": bool(required) and any(r.decision == "reject" for r in counted),
        "missing_role_coverage": bool(required) and not required <= covered,
        "too_few_distinct_reviewers": bool(required) and
            len({r.reviewer for r in approvals}) < int(contract["min_reviewers"]),
    }


def check(spec: dict) -> Tuple[bool, Optional[str]]:
    """Decision for the scenario's last release event."""
    contract = spec.get("contract")
    registry = {rid: frozenset(roles) for rid, roles in spec.get("reviewers", {}).items()}
    reviews: List[_Review] = []
    released = False
    result: Tuple[bool, Optional[str]] = (False, None)

    for ev in spec["events"]:
        kind = ev["type"]
        if kind == "review" and contract is not None:
            reviews.append(_Review(ev["reviewer"], registry.get(ev["reviewer"], frozenset()),
                                   ev["decision"], float(ev["time"])))
        elif kind == "register_reviewer":
            registry[ev["reviewer"]] = frozenset(ev["roles"])
        elif kind == "revoke_reviewer":
            registry.pop(ev["reviewer"], None)
        elif kind == "release":
            failed = _failures(contract, released, reviews, registry, ev)
            first = next((r for r in REASON_ORDER if failed.get(r)), None)
            result = (first is None, first)
            released = released or first is None
    return result


def ordinal_only(spec: dict) -> Tuple[bool, Optional[str]]:
    """Permit iff the last release's action meets the floor (1 if no contract)."""
    contract = spec.get("contract")
    floor = int(contract["floor"]) if contract is not None else 1
    releases = [ev for ev in spec["events"] if ev["type"] == "release"]
    if int(releases[-1]["action"]) >= floor:
        return True, None
    return False, "action_below_floor"
