"""
20 workflow scenarios x 4 namespace labels = 80 cases.

Scenarios are plain data (the same schema that halo.intake reads from JSON
files), so three independent evaluators can run them:

- the gate (``run_on_gate``), which replays the events through ReleaseGate;
- the static contract checker (``halo.checker.check``), which evaluates the
  release condition directly from the scenario fields;
- the ordinal-only control (``halo.checker.ordinal_only``), which checks
  only the floor of Equation 1.

Schema
------
{
  "name": str,
  "expected": {"permit": bool, "reason": str | null},   # optional for intake
  "reviewers": {reviewer_id: [roles...]},                # initial registry
  "contract": null | {                                   # null: never issued
      "output": str, "rule_revision": str, "obs_revision": str,
      "required_roles": [roles...], "min_reviewers": int, "floor": int,
      "window_start": float, "deadline": float},
  "events": [                                            # in order
      {"type": "review", "reviewer": id, "decision": "approve"|"reject", "time": float},
      {"type": "register_reviewer", "reviewer": id, "roles": [roles...]},
      {"type": "revoke_reviewer", "reviewer": id},
      {"type": "release", "output": str, "rule_revision": str,
       "obs_revision": str, "time": float, "action": int}]
}
The scenario's decision is the result of its last release event.

Labels change no mechanism; they are namespace prefixes for case IDs.
"""

from typing import Dict, List, Optional, Tuple

from ..gate import Contract, ReleaseGate, output_hash


LABELS = ["clinical", "parenting", "eldercare", "mental_health"]


def _contract(output, roles, floor=5, window=(0.0, 100.0), k=1):
    return {"output": output, "rule_revision": "v1", "obs_revision": "o1",
            "required_roles": list(roles), "min_reviewers": k, "floor": floor,
            "window_start": window[0], "deadline": window[1]}


def _review(reviewer, decision, time):
    return {"type": "review", "reviewer": reviewer, "decision": decision, "time": time}


def _release(output, time, action=5, rule_revision="v1", obs_revision="o1"):
    return {"type": "release", "output": output, "rule_revision": rule_revision,
            "obs_revision": obs_revision, "time": time, "action": action}


def _permit():
    return {"permit": True, "reason": None}


def _hold(reason):
    return {"permit": False, "reason": reason}


SCENARIOS: List[dict] = [
    {"name": "valid_general_review",
     "expected": _permit(),
     "reviewers": {"rev1": ["general"]},
     "contract": _contract("Output for general review", ["general"]),
     "events": [_review("rev1", "approve", 10.0),
                _release("Output for general review", 50.0)]},

    {"name": "valid_specialist_review",
     "expected": _permit(),
     "reviewers": {"pharm1": ["pharmacist"]},
     "contract": _contract("Specialist output", ["pharmacist"]),
     "events": [_review("pharm1", "approve", 20.0),
                _release("Specialist output", 50.0)]},

    {"name": "missing_review",
     "expected": _hold("missing_role_coverage"),
     "reviewers": {"rev1": ["general"]},
     "contract": _contract("Unreviewed output", ["general"]),
     "events": [_release("Unreviewed output", 50.0)]},

    # A reviewer whose role does not match the contract rejects; a qualified
    # reviewer approves.
    {"name": "nonqualified_rejection",
     "expected": _permit(),
     "reviewers": {"rev1": ["nurse"], "rev2": ["general"]},
     "contract": _contract("Non-qualified rejection", ["general"]),
     "events": [_review("rev1", "reject", 10.0),
                _review("rev2", "approve", 15.0),
                _release("Non-qualified rejection", 50.0)]},

    # Review completed after the deadline; the release is also after the
    # deadline, so the gate holds on the release window first.
    {"name": "late_review",
     "expected": _hold("outside_release_window"),
     "reviewers": {"rev1": ["general"]},
     "contract": _contract("Late review output", ["general"], window=(0.0, 50.0)),
     "events": [_review("rev1", "approve", 60.0),
                _release("Late review output", 70.0)]},

    # Review timestamped before the window opens (named "future_dated_review"
    # in the submitted artifact).
    {"name": "pre_window_review",
     "expected": _hold("missing_role_coverage"),
     "reviewers": {"rev1": ["general"]},
     "contract": _contract("Future dated", ["general"], window=(10.0, 100.0)),
     "events": [_review("rev1", "approve", 5.0),
                _release("Future dated", 50.0)]},

    {"name": "wrong_role_review",
     "expected": _hold("missing_role_coverage"),
     "reviewers": {"rev1": ["nurse"]},
     "contract": _contract("Wrong role output", ["pharmacist"]),
     "events": [_review("rev1", "approve", 10.0),
                _release("Wrong role output", 50.0)]},

    {"name": "expired_release_window",
     "expected": _hold("outside_release_window"),
     "reviewers": {"rev1": ["general"]},
     "contract": _contract("Expired window", ["general"], window=(0.0, 50.0)),
     "events": [_review("rev1", "approve", 10.0),
                _release("Expired window", 60.0)]},

    {"name": "changed_output",
     "expected": _hold("payload_changed"),
     "reviewers": {"rev1": ["general"]},
     "contract": _contract("Original output", ["general"]),
     "events": [_review("rev1", "approve", 10.0),
                _release("Regenerated output", 50.0)]},

    {"name": "changed_rules",
     "expected": _hold("rules_changed"),
     "reviewers": {"rev1": ["general"]},
     "contract": _contract("Rules changed output", ["general"]),
     "events": [_review("rev1", "approve", 10.0),
                _release("Rules changed output", 50.0, rule_revision="v2")]},

    {"name": "changed_observations",
     "expected": _hold("observations_changed"),
     "reviewers": {"rev1": ["general"]},
     "contract": _contract("Obs changed output", ["general"]),
     "events": [_review("rev1", "approve", 10.0),
                _release("Obs changed output", 50.0, obs_revision="o2")]},

    {"name": "revoked_reviewer",
     "expected": _hold("missing_role_coverage"),
     "reviewers": {"rev1": ["general"]},
     "contract": _contract("Revoked reviewer", ["general"]),
     "events": [_review("rev1", "approve", 10.0),
                {"type": "revoke_reviewer", "reviewer": "rev1"},
                _release("Revoked reviewer", 50.0)]},

    {"name": "replay_release",
     "expected": _hold("already_released"),
     "reviewers": {"rev1": ["general"]},
     "contract": _contract("Replay output", ["general"]),
     "events": [_review("rev1", "approve", 10.0),
                _release("Replay output", 50.0),
                _release("Replay output", 55.0)]},

    {"name": "incomplete_role_coverage",
     "expected": _hold("missing_role_coverage"),
     "reviewers": {"rev1": ["general"], "rev2": ["pharmacist"]},
     "contract": _contract("Incomplete coverage", ["general", "pharmacist"]),
     "events": [_review("rev1", "approve", 10.0),
                _release("Incomplete coverage", 50.0)]},

    {"name": "action_below_floor",
     "expected": _hold("action_below_floor"),
     "reviewers": {"rev1": ["general"]},
     "contract": _contract("Below floor", ["general"]),
     "events": [_review("rev1", "approve", 10.0),
                _release("Below floor", 50.0, action=3)]},

    {"name": "no_roles_required",
     "expected": _permit(),
     "reviewers": {},
     "contract": _contract("No roles needed", [], floor=1),
     "events": [_release("No roles needed", 50.0, action=1)]},

    # The hazard is not encoded in any obligation, so the gate permits by design.
    {"name": "unencoded_hazard",
     "expected": _permit(),
     "reviewers": {},
     "contract": _contract("Unencoded hazard output", [], floor=1),
     "events": [_release("Unencoded hazard output", 50.0, action=1)]},

    {"name": "two_roles_two_reviewers",
     "expected": _permit(),
     "reviewers": {"rev1": ["general"], "rev2": ["pharmacist"]},
     "contract": _contract("Two roles OK", ["general", "pharmacist"]),
     "events": [_review("rev1", "approve", 10.0),
                _review("rev2", "approve", 15.0),
                _release("Two roles OK", 50.0)]},

    {"name": "never_issued",
     "expected": _hold("unknown_case"),
     "reviewers": {},
     "contract": None,
     "events": [_release("Never issued output", 50.0)]},

    {"name": "multi_role_single_reviewer",
     "expected": _permit(),
     "reviewers": {"rev1": ["general", "pharmacist"]},
     "contract": _contract("Multi role single reviewer", ["general", "pharmacist"]),
     "events": [_review("rev1", "approve", 10.0),
                _release("Multi role single reviewer", 50.0)]},
]

assert len(SCENARIOS) == 20, f"Expected 20 scenarios, got {len(SCENARIOS)}"


def run_on_gate(spec: dict, case_id: str,
                gate: Optional[ReleaseGate] = None) -> Tuple[bool, Optional[str]]:
    """Replay a scenario's events through a gate. Returns the last release
    decision as (permit, reason name or None)."""
    gate = gate if gate is not None else ReleaseGate()
    for rid, roles in spec.get("reviewers", {}).items():
        gate.register_reviewer(rid, set(roles))
    c = spec.get("contract")
    if c is not None:
        gate.issue(Contract(
            case_id, output_hash(c["output"]), c["rule_revision"], c["obs_revision"],
            frozenset(c["required_roles"]), int(c["min_reviewers"]), int(c["floor"]),
            float(c["window_start"]), float(c["deadline"]),
        ))
    result: Tuple[bool, Optional[str]] = (False, None)
    for ev in spec["events"]:
        kind = ev["type"]
        if kind == "review":
            gate.record_review(case_id, ev["reviewer"], ev["decision"], float(ev["time"]))
        elif kind == "register_reviewer":
            gate.register_reviewer(ev["reviewer"], set(ev["roles"]))
        elif kind == "revoke_reviewer":
            gate.revoke_reviewer(ev["reviewer"])
        elif kind == "release":
            permit, reason = gate.release(
                case_id, output_hash(ev["output"]), ev["rule_revision"],
                ev["obs_revision"], float(ev["time"]), int(ev["action"]),
            )
            result = (permit, reason.value if reason is not None else None)
        else:
            raise ValueError(f"unknown event type: {kind}")
    return result


def scenario_cases() -> List[Tuple[str, dict]]:
    """All 80 (case_id, scenario) pairs: 20 scenarios under 4 labels."""
    return [(f"{label}_s{idx:02d}", spec)
            for label in LABELS
            for idx, spec in enumerate(SCENARIOS)]


def run_all_scenarios() -> List[Dict]:
    """Run all 80 cases on the gate."""
    rows = []
    for case_id, spec in scenario_cases():
        permit, reason = run_on_gate(spec, case_id)
        rows.append({"case_id": case_id, "name": spec["name"],
                     "expected_permit": spec["expected"]["permit"],
                     "expected_reason": spec["expected"]["reason"],
                     "permit": permit, "reason": reason})
    return rows
